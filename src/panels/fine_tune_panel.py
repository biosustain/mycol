import numpy as np
import pandas as pd
import streamlit as st

from src.helpers.grid_helpers import render_image_selection_table
from src.helpers.state_ops import selected_training_keys
from src.helpers.densenet_functions import (
    build_densenet_zip_bytes,
    start_densenet_training,
    check_densenet_training_status,
    cancel_densenet_training,
)
from src.helpers.cellpose_functions import (
    DEFAULT_CELLPOSE_MODEL,
    preprocess_for_cellpose,
    start_cellpose_training,
    check_cellpose_training_status,
    cancel_cellpose_training,
    start_cellpose_validation,
    check_cellpose_validation_status,
    cancel_cellpose_validation,
)
from src.helpers.help_panels import (
    classifier_training_plot_help,
    cellpose_training_plot_help,
)

ss = st.session_state


# ========== DenseNet: options (light) + dataset summary (light-ish) + training (heavy) ==========


def render_densenet_image_selection():
    """Step 2 (classifier): choose which images contribute labelled cells."""
    st.caption("Only labelled cells from the selected images are used.")
    return render_image_selection_table("dn")


def render_densenet_params():
    """Step 3 (classifier): training hyperparameters."""
    c1, c2, c4 = st.columns(3)
    # options for densenet training with defaults already set
    ss["dn_input_size"] = c1.selectbox(
        "Input size",
        options=[64, 96, 128],
        index=[64, 96, 128].index(ss["dn_input_size"]),
    )
    ss["dn_batch_size"] = c2.selectbox(
        "Batch size",
        options=[8, 16, 32, 64],
        index=[8, 16, 32, 64].index(ss["dn_batch_size"]),
    )

    ss["dn_max_epoch"] = c4.number_input(
        "Max epochs",
        min_value=1,
        max_value=500,
        value=int(ss["dn_max_epoch"]),
        step=5,
        key="max_epoch_densenet_ui",
    )


def _densenet_class_counts():
    """Per-class labelled-cell counts for the DenseNet training set, from label dicts."""
    classes = [c for c in ss.get("all_classes", []) if c != "No label"] or ["class0", "class1"]
    tally = {}
    for k in selected_training_keys("dn"):
        rec = ss["images"][k]
        M, labs = rec.get("masks"), rec.get("labels", {})
        if not isinstance(M, np.ndarray) or not M.any():
            continue
        for i in np.unique(M):
            name = labs.get(int(i))
            tally[name] = tally.get(name, 0) + 1
    return np.array([tally.get(c, 0) for c in classes], dtype=int), classes


@st.fragment
def render_densenet_summary_fragment():
    """Show a simple class frequency table (counts read from label dicts, no patches)."""
    counts, classes = _densenet_class_counts()
    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.info(
        f"Training set: {int(counts.sum())} labelled cells across {len(classes)} classes.",
        width="stretch",
    )
    st.dataframe(
        freq_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
        },
    )


def densenet_can_train(min_classes: int = 2, min_instances: int = 2) -> bool:
    """True if at least `min_classes` classes have >= `min_instances` labelled cells each."""
    counts, _ = _densenet_class_counts()
    return int((counts >= min_instances).sum()) >= min_classes


def render_densenet_train_fragment():
    """Runs the full DenseNet training pipeline when the button is clicked."""

    dn_job = ss.get("dn_training_job")
    cp_job = ss.get("cp_training_job")

    # Read hyperparameter options from session
    input_size = int(ss.get("dn_input_size"))
    batch_size = int(ss.get("dn_batch_size"))
    epochs = int(ss.get("dn_max_epoch"))
    val_split = 0.2

    # --- check if we have enough data to train ---
    can_train = densenet_can_train()

    if not can_train:
        st.warning(
            "Need at least 2 classes with ≥ 2 labelled cells each "
            "before fine-tuning DenseNet."
        )

    # Check for active training job
    status_container = st.empty()
    with status_container:
        render_densenet_status_fragment()

    if dn_job and dn_job.get("status") == "running":
        return

    # Disable button if we don't have enough data or another job is running
    other_job_running = bool(cp_job and cp_job.get("status") == "running")
    button_disabled = not can_train or other_job_running

    if other_job_running:
        st.warning(
            "⚠️ Cellpose training is currently running. Wait for it to finish before starting DenseNet training."
        )

    # training-set summary + class breakdown, shown just above the fine-tune button
    render_densenet_summary_fragment()

    go = st.button(
        "Fine tune Densenet121",
        width="stretch",
        type="primary",
        disabled=button_disabled,
    )

    # Don't proceed if button not clicked or training is not allowed
    if not go or button_disabled:
        return

    # Start async training
    start_densenet_training(
        input_size=input_size,
        batch_size=batch_size,
        epochs=epochs,
        val_split=val_split,
    )
    st.rerun()


def show_densenet_training_plots():
    """Render saved DenseNet training plots (only reached once a model is trained)."""
    classifier_training_plot_help()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            ss["densenet_training_losses"],
            width="stretch",
        )
    with col2:
        st.plotly_chart(
            ss["densenet_training_metrics"],
            width="stretch",
        )

    st.plotly_chart(
        ss["densenet_confusion_matrix"],
        width="stretch",
    )


# ========== Cellpose: options + training ==========


def render_cellpose_image_selection():
    """Step 2 (segmenter): choose which images contribute masks."""
    st.caption("Only masks from the selected images are used.")
    return render_image_selection_table("cp")


def render_cellpose_params():
    """Step 3 (segmenter): training hyperparameters."""
    # --- show training options ---
    c1, c2, c3 = st.columns(3)

    # Learning rate and weight decay are fixed by the Cellpose-SAM protocol and live
    # in the session-state defaults (see state_ops), so they are not exposed here.
    ss["cp_base_model"] = DEFAULT_CELLPOSE_MODEL  # Cellpose-SAM is the only base model
    ss.setdefault("cp_max_epoch", 100)
    ss.setdefault("cp_batch_size", 1)
    ss.setdefault("cp_nimg_per_epoch", 8)
    ss.setdefault("cp_min_cells_per_image", 5)

    ss["cp_max_epoch"] = c2.number_input(
        "Max epochs",
        1,
        1000,
        int(ss["cp_max_epoch"]),
        step=10,
        help="Number of training epochs for fine-tuning. Longer training may improve performance but takes more time.",
    )

    ss["cp_min_cells_per_image"] = c2.number_input(
        "Min cells per image",
        min_value=1,
        max_value=10000,
        value=int(ss["cp_min_cells_per_image"]),
        step=1,
        key="cp_min_cells_per_image_input",
        help="Images with fewer than this many annotated cells are excluded from training. Cellpose default is 5.",
    )

    BATCH_OPTS = [1, 2, 4]
    ss["cp_batch_size"] = c3.selectbox(
        "Batch size",
        options=BATCH_OPTS,
        index=BATCH_OPTS.index(ss["cp_batch_size"])
        if ss["cp_batch_size"] in BATCH_OPTS
        else 0,
        key="cellpose_batch_size",
        help="Images per training batch. Cellpose-SAM is a 305M-parameter transformer, so "
        "training memory is the binding constraint — 1 is the value the paper uses and the "
        "only one guaranteed to fit on a laptop GPU.",
    )

    NIMG_OPTS = [None, 8, 32, 64, 128, 256]
    ss["cp_nimg_per_epoch"] = c2.selectbox(
        "Images per epoch",
        options=NIMG_OPTS,
        index=NIMG_OPTS.index(ss["cp_nimg_per_epoch"])
        if ss["cp_nimg_per_epoch"] in NIMG_OPTS
        else 0,
        format_func=lambda v: "All" if v is None else str(v),
        key="cellpose_nimg_per_epoch",
        help="How many training images to sample each epoch. 'All' uses every training image.",
    )

    # --- Hyperparameter tuning toggle + grid inputs ---
    with c1:
        subcol1, subcol2 = st.columns([2, 3])
        with subcol1:

            ss.setdefault("cp_do_gridsearch", True)
            ss["cp_do_gridsearch"] = st.checkbox(
                "Optimise hyperparameters",
                value=bool(ss["cp_do_gridsearch"]),
                help="Gain a small performance bonus by also optmising Cellpose hyperparameters. Remember to set these when using the fine-tuned model later in the 'Annotate Images' page.",
            )

        with subcol2:
            if ss["cp_do_gridsearch"]:
                ss["cp_n_trials"] = st.slider(
                    min_value=10,
                    max_value=60,
                    value=int(ss.get("cp_n_trials", 20)),
                    step=5,
                    label="Optimisation iterations",
                    help="Number of hyperparameter combinations to try during optimisation. More trials may yield better results but take longer.",
                )


def render_cellpose_summary():
    """Training-set summary for the selected Cellpose images (filtered by min cells per image)."""

    def is_mask(m):
        return isinstance(m, np.ndarray) and m.ndim == 2 and m.any()

    min_cells = int(ss.get("cp_min_cells_per_image", 5))
    n_pass_images, n_pass_masks = 0, 0
    n_fail_images = 0
    for k in selected_training_keys("cp"):
        rec = ss["images"][k]
        m = rec["masks"]
        n = int(len(np.unique(m)) - 1) if is_mask(m) else 0
        if n >= min_cells:
            n_pass_images += 1
            n_pass_masks += n
        else:
            n_fail_images += 1

    st.info(
        f"Training set: {n_pass_masks} cell masks across {n_pass_images} images "
        f"(≥ {min_cells} cells per image).",
        width="stretch",
    )
    if n_fail_images > 0:
        st.info(
            f"{n_fail_images} image{'s' if n_fail_images != 1 else ''} excluded "
            f"(fewer than {min_cells} annotated cells).",
            width="stretch",
        )


def get_train_setup():
    min_cells = int(ss.get("cp_min_cells_per_image", 5))
    recs = {
        k: ss["images"][k]
        for k in selected_training_keys("cp")
        if int(len(np.unique(ss["images"][k]["masks"])) - 1) >= min_cells
    }
    base_model = ss.get("cp_base_model")
    epochs = int(ss.get("cp_max_epoch"))
    lr = float(ss.get("cp_learning_rate"))
    wd = float(ss.get("cp_weight_decay"))
    batch_size = int(ss.get("cp_batch_size"))
    nimg_per_epoch = ss.get("cp_nimg_per_epoch")  # None = all training images
    return recs, base_model, epochs, lr, wd, batch_size, nimg_per_epoch, min_cells


@st.cache_data(show_spinner=False)
def prepare_eval_data(recs):
    """Preprocess all images in recs order (matches the finetune worker's order)."""
    names = [rec.get("name", f"Image {i}") for i, rec in enumerate(recs.values())]
    masks = [rec["masks"] for rec in recs.values()]
    images = [preprocess_for_cellpose(rec) for rec in recs.values()]
    return images, masks, names


def render_cellpose_train_fragment():
    """Runs the full Cellpose fine-tuning pipeline when the button is clicked."""

    # Check for any active training job
    cp_job = ss.get("cp_training_job")
    dn_job = ss.get("dn_training_job")

    # Check for active training/validation jobs
    val_job = ss.get("cp_validation_job")
    status_container = st.empty()
    with status_container:
        render_cellpose_status_fragment()

    if (cp_job and cp_job.get("status") == "running") or (
        val_job and val_job.get("status") == "running"
    ):
        return

    # Disable if another job is running
    other_job_running = bool(dn_job and dn_job.get("status") == "running")

    if other_job_running:
        st.warning(
            "⚠️ DenseNet training is currently running. Wait for it to finish before starting Cellpose training."
        )

    # training-set summary, shown just above the fine-tune button
    render_cellpose_summary()

    go = st.button(
        "Fine-tune Cellpose",
        width="stretch",
        type="primary",
        disabled=other_job_running,
    )
    if not go:
        return

    # Start async training
    recs, base_model, epochs, lr, wd, batch_size, nimg_per_epoch, min_cells = (
        get_train_setup()
    )
    start_cellpose_training(
        recs, base_model, epochs, lr, wd, batch_size, nimg_per_epoch,
        min_train_masks=min_cells,
    )
    st.rerun()


def show_cellpose_training_plots():
    """Render saved Cellpose plots (only reached once a model is trained)."""
    cellpose_training_plot_help()

    col1, col2 = st.columns(2)
    with col1:

        # plot training losses
        st.plotly_chart(
            ss["cellpose_training_losses"],
            width="stretch",
        )

        # plot original vs predicted counts
        if "cellpose_original_counts_comparison" in ss:
            st.plotly_chart(
                ss["cellpose_original_counts_comparison"],
                width="stretch",
            )

    with col2:

        # plot iou comparison
        if "cellpose_iou_comparison" in ss:
            st.plotly_chart(
                ss["cellpose_iou_comparison"],
                width="stretch",
            )

        # plot tuned vs predicted counts
        if "cellpose_tuned_counts_comparison" in ss:
            st.plotly_chart(
                ss["cellpose_tuned_counts_comparison"],
                width="stretch",
            )

    # display grid search results if applicable
    if ss.get("cp_do_gridsearch") and "cp_grid_results_df" in ss:
        st.subheader("Tested hyperparameters")
        st.dataframe(
            ss["cp_grid_results_df"],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No hyperparameter tuning performed.")


def _tick_icon(toggle_key: str) -> str:
    """Return an alternating hourglass icon, flipping the stored toggle each call."""
    ss.setdefault(toggle_key, True)
    icon = "⌛" if ss[toggle_key] else "⏳"
    ss[toggle_key] = not ss[toggle_key]
    return icon


def _render_running_status(
    job: dict,
    *,
    icon: str,
    running_msg: str,
    noun: str,
    cancel_label: str,
    cancel_key: str,
    cancel_fn,
) -> None:
    """Shared 'running' status UI: progress line, log viewer, and cancel button."""
    st.info(f"{icon} {running_msg}")

    log_content = job.get("log_content", "")
    if log_content:
        with st.expander(f"📄 View {noun.title()} Log", expanded=False):
            st.code(log_content, language="text")
    else:
        st.caption(f"Waiting for {noun} output...")

    if st.button(cancel_label, type="primary", key=cancel_key, width="stretch"):
        cancel_fn()
        st.rerun()


def _render_failed_status(job: dict, error_msg: str) -> None:
    """Shared 'failed' status UI: error line with collapsible worker log."""
    st.error(error_msg)
    with st.expander("Show Error Details"):
        st.code(job.get("error", "Unknown error"))


@st.fragment(run_every=2)
def render_densenet_status_fragment():
    """Real-time status and log viewer for DenseNet training."""
    dn_job = ss.get("dn_training_job")
    if not dn_job or dn_job.get("status") != "running":
        return

    status = check_densenet_training_status()
    icon = _tick_icon("dn_icon_toggle")

    if status == "running":
        _render_running_status(
            dn_job,
            icon=icon,
            running_msg="DenseNet training in progress...",
            noun="training",
            cancel_label="Cancel Training",
            cancel_key="cancel_dn_frag",
            cancel_fn=cancel_densenet_training,
        )
    elif status == "complete":
        st.success("Finalizing DenseNet training...")
        st.balloons()  # cool as f*ck

        ss["dn_zip_bytes"] = build_densenet_zip_bytes(dn_job["input_size"])
        ss.pop("dn_training_job", None)
        st.rerun()
    elif status == "failed":
        _render_failed_status(dn_job, "❌ DenseNet training failed.")
        # ss.pop("dn_training_job", None)
        # st.rerun()


@st.fragment(run_every=10)
def render_cellpose_status_fragment():
    """Real-time status and log viewer for Cellpose training and validation"""
    cp_job = ss.get("cp_training_job")
    val_job = ss.get("cp_validation_job")

    from src.helpers.cellpose_functions import (
        build_cellpose_zip_bytes,
    )

    icon = _tick_icon("cp_icon_toggle")

    # Check Training
    if cp_job and cp_job.get("status") == "running":
        status = check_cellpose_training_status()

        if status == "running":
            _render_running_status(
                cp_job,
                icon=icon,
                running_msg="Cellpose training in progress...",
                noun="training",
                cancel_label="Cancel Training",
                cancel_key="cancel_cp_frag",
                cancel_fn=cancel_cellpose_training,
            )
        elif status == "complete":
            st.success(
                "Finalizing Cellpose training..."
            )  # technically its done but the synchronous parts here are too slow

            from src.panels.fine_tune_panel import get_train_setup

            recs, base_model, epochs, lr, wd, batch_size, nimg_per_epoch, _ = (
                get_train_setup()
            )
            start_cellpose_validation(
                recs=recs,
                base_model=base_model,
                do_gridsearch=ss.get("cp_do_gridsearch", False),
                n_trials=ss.get("cp_n_trials", 20),
            )
            ss.pop("cp_training_job", None)
            st.rerun()
        elif status == "failed":
            _render_failed_status(cp_job, "❌ Cellpose training failed.")
            # ss.pop("cp_training_job", None)
            # st.rerun()
        return

    # Check Validation
    if val_job and val_job.get("status") == "running":
        val_status = check_cellpose_validation_status()

        if val_status == "running":
            _render_running_status(
                val_job,
                icon=icon,
                running_msg="Cellpose validation in progress...",
                noun="validation",
                cancel_label="Cancel Validation",
                cancel_key="cancel_val_frag",
                cancel_fn=cancel_cellpose_validation,
            )
        elif val_status == "complete":
            st.success("Finalizing validation... please stay on this page for now!")
            st.balloons()
            ss["cp_zip_bytes"] = build_cellpose_zip_bytes()
            ss.pop("cp_validation_job", None)
            st.rerun()
        elif val_status == "failed":
            _render_failed_status(val_job, "❌ Validation failed.")
            # ss.pop("cp_validation_job", None)
            # st.rerun()
