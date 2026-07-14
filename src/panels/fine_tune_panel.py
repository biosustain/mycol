# panels/train_densenet.py
import numpy as np
import pandas as pd
import streamlit as st

from cellpose import metrics, core
import optuna

from src.helpers.state_ops import ordered_keys, plot_loss_curve
from src.helpers.densenet_functions import (
    load_labeled_patches,
    build_densenet_zip_bytes,
    start_densenet_training,
    check_densenet_training_status,
    cancel_densenet_training,
)
from src.helpers.cellpose_functions import (
    compute_prediction_ious,
    plot_iou_comparison,
    plot_pred_vs_true_counts,
    get_cellpose_model,
    get_tuned_model,
    build_cellpose_zip_bytes,
    start_cellpose_training,
    check_cellpose_training_status,
    cancel_cellpose_training,
    start_cellpose_validation,
    check_cellpose_validation_status,
    cancel_cellpose_validation,
    CellposeModel3Proxy,
)
from src.helpers.help_panels import (
    classifier_training_plot_help,
    cellpose_training_plot_help,
)

ss = st.session_state


# ========== DenseNet: options (light) + dataset summary (light-ish) + training (heavy) ==========


@st.fragment
def render_densenet_options(key_ns="train_densenet"):
    """Light controls - lives outside fragments so changing options refreshes summary."""
    st.header("Fine-tune a DenseNet classifier")

    # show information about the training set
    render_densenet_summary_fragment()

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
    return True


@st.fragment
def render_densenet_summary_fragment():
    """Loads patches and shows a simple class frequency table (reruns when the page reruns)."""
    input_size = int(ss.get("dn_input_size"))

    # Load patches only for summary; heavy-ish but isolated here
    X, y, classes = load_labeled_patches(patch_size=input_size)

    # Count occurrences per class (ensure all classes present)
    counts = np.bincount(y, minlength=len(classes))
    freq_df = pd.DataFrame({"Class": list(classes), "Count": counts.astype(int)})

    st.info(
        f"Training set: {int(counts.sum())} labelled cells across {len(classes)} classes."
    )
    # Pretty, form-like card with rounded edges
    st.dataframe(
        freq_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Class": st.column_config.TextColumn("Class"),
            "Count": st.column_config.NumberColumn("Count", format="%d"),
        },
    )


def densenet_can_train(
    patch_size: int, min_classes: int = 2, min_instances: int = 2
) -> bool:
    """
    Return True if there are at least `min_classes` classes
    with >= `min_instances` examples each.
    """
    _, y, classes = load_labeled_patches(patch_size=patch_size)

    # counts[i] = number of samples for class i
    counts = np.bincount(y, minlength=len(classes))

    # how many classes have at least `min_instances` examples?
    n_ok = int((counts >= min_instances).sum())
    return n_ok >= min_classes


def render_densenet_train_fragment():
    """Runs the full DenseNet training pipeline when the button is clicked."""

    dn_job = ss.get("dn_training_job")
    cp_job = ss.get("cp_training_job")
    (dn_job and dn_job.get("status") == "running") or (
        cp_job and cp_job.get("status") == "running"
    )

    # Read hyperparameter options from session
    input_size = int(ss.get("dn_input_size"))
    batch_size = int(ss.get("dn_batch_size"))
    epochs = int(ss.get("dn_max_epoch"))
    val_split = 0.2

    # --- check if we have enough data to train ---
    can_train = densenet_can_train(patch_size=input_size)

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
    """Render saved DenseNet training plots from session state (if available)."""

    # check for uploaded data
    k1, k2 = "densenet_training_metrics", "densenet_training_metrics"
    if (k1 not in st.session_state) and (k2 not in st.session_state):
        st.info("No fine-tuning data available. Tune a model first.")
        st.empty()
        return

    else:

        st.header("DenseNet Training Summary")

        classifier_training_plot_help()

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                st.session_state["densenet_training_losses"],
                width="stretch",
            )
        with col2:
            st.plotly_chart(
                st.session_state["densenet_training_metrics"],
                width="stretch",
            )

        st.plotly_chart(
            st.session_state["densenet_confusion_matrix"],
            width="stretch",
        )


# ========== Cellpose: options + training ==========


def render_cellpose_options(key_ns="train_cellpose"):
    st.header("Fine-tune a Cellpose segmenter")

    # --- show training options ---
    c1, c2, c3 = st.columns(3)

    # Defaults
    ss.setdefault("cp_base_model", "cyto2")
    ss.setdefault("cp_max_epoch", 100)
    ss.setdefault("cp_learning_rate", 0.1)
    ss.setdefault("cp_weight_decay", 1e-4)
    ss.setdefault("cp_batch_size", 32)
    ss.setdefault("cp_min_cells_per_image", 5)

    ss["cp_base_model"] = c1.selectbox(
        "Base model",
        options=["cyto", "cyto2", "cyto3", "nuclei", "scratch"],
        index=["cyto", "cyto2", "cyto3", "nuclei", "scratch"].index(
            ss["cp_base_model"]
        ),  # this line sets the
    )
    ss["cp_max_epoch"] = c2.number_input(
        "Max epochs",
        1,
        1000,
        int(ss["cp_max_epoch"]),
        step=10,
        help="Number of training epochs for fine-tuning. Longer training may improve performance but takes more time.",
    )
    ss["cp_learning_rate"] = c3.number_input(
        "Learning rate",
        min_value=1e-8,
        max_value=10.0,
        value=float(ss["cp_learning_rate"]),
        format="%.5f",
        help="Initial learning rate for the optimizer. Lower values may lead to more stable training.",
    )
    ss["cp_weight_decay"] = c1.number_input(
        "Weight decay",
        min_value=0.0,
        max_value=1.0,
        value=float(ss["cp_weight_decay"]),
        step=1e-8,
        format="%.8f",  # more decimals prevents snapping to 0
        key="cp_weight_decay_input",
        help="Weight decay over the last epochs of training.",
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

    ss["cp_batch_size"] = c2.selectbox(
        "Batch size",
        options=[8, 16, 32, 64],
        index=[8, 16, 32, 64].index(ss["cp_batch_size"]),
        key="cellpose_batch_size",
        help="Number of images per training batch. Larger batch sizes speed up training but require more memory.",
    )

    # --- Hyperparameter tuning toggle + grid inputs ---
    with c1:
        subcol1, subcol2 = st.columns([2, 3])
        with subcol1:

            ss.setdefault("cp_do_gridsearch", False)
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
                    value=20,
                    step=5,
                    label="Optimisation iterations",
                    help="Number of hyperparameter combinations to try during optimisation. More trials may yield better results but take longer.",
                )

    # --- show dataset stats filtered by min_cells_per_image ---
    def is_mask(m):
        return isinstance(m, np.ndarray) and m.ndim == 2 and m.any()

    min_cells = int(ss["cp_min_cells_per_image"])
    n_pass_images, n_pass_masks = 0, 0
    n_fail_images = 0
    for k in ordered_keys():
        rec = st.session_state["images"][k]
        m = rec["masks"]
        n = int(len(np.unique(m)) - 1) if is_mask(m) else 0
        if n >= min_cells:
            n_pass_images += 1
            n_pass_masks += n
        else:
            n_fail_images += 1

    info_col1, info_col2 = st.columns(2)
    info_col1.info(
        f"Training set: {n_pass_masks} cell masks across {n_pass_images} images "
        f"(≥ {min_cells} cells per image)."
    )
    if n_fail_images > 0:
        info_col2.info(
            f"{n_fail_images} image{'s' if n_fail_images != 1 else ''} excluded "
            f"(fewer than {min_cells} annotated cells)."
        )


def get_train_setup():
    min_cells = int(ss.get("cp_min_cells_per_image", 5))
    recs = {
        k: st.session_state["images"][k]
        for k in ordered_keys()
        if int(len(np.unique(st.session_state["images"][k]["masks"])) - 1) >= min_cells
    }
    base_model = ss.get("cp_base_model")
    epochs = int(ss.get("cp_max_epoch"))
    lr = float(ss.get("cp_learning_rate"))
    wd = float(ss.get("cp_weight_decay"))
    nimg = int(ss.get("cp_batch_size"))
    # images are converted to grayscale in preprocess_for_cellpose, so channels are fixed
    channels = [0, 0]
    return recs, base_model, epochs, lr, wd, nimg, channels, min_cells


@st.cache_data(show_spinner=False)
def prepare_eval_data(recs, max_n=40):
    """returns a random subset of data on which to perform hyperparameter tuning"""
    names = [rec.get("name", f"Image {i}") for i, rec in enumerate(recs.values())]
    masks = [rec["masks"] for rec in recs.values()]
    images = [rec["image"] for rec in recs.values()]
    N = len(images)
    sample_n = min(max_n, N)
    if N > sample_n:
        rng = np.random.default_rng()
        idx = rng.choice(N, size=sample_n, replace=False)
        images = [images[i] for i in idx]
        masks = [masks[i] for i in idx]
        names = [names[i] for i in idx]
    return images, masks, names


def set_cp_hparams(src):
    ss["cp_cellprob_threshold"] = float(src["cellprob"])
    ss["cp_flow_threshold"] = float(src["flow_threshold"])
    ss["cp_min_size"] = int(src["min_size"])
    ss["cp_niter"] = int(src["niter"])


def run_optuna(images, masks, base_model, channels, model_name):
    if not ss.get("cp_do_gridsearch"):
        return channels

    st.subheader("Hyperparameter tuning (Optuna)")

    eval_model = get_tuned_model()

    pb = st.progress(0.0, text="Starting Optuna optimisation…")
    results = []
    n_trials = st.session_state.get("cp_n_trials")

    def objective(trial: optuna.trial.Trial) -> float:
        cellprob = trial.suggest_float("cellprob", -0.2, 0.6, step=0.2)
        flowthresh = trial.suggest_float("flow_threshold", -0.2, 0.6, step=0.2)
        niter = trial.suggest_int("niter", 100, 1000, step=100)
        min_size = trial.suggest_int("min_size", 0, 500, step=50)

        i = trial.number + 1
        pb.progress(
            min(i / n_trials, 1.0),
            text=(
                f"Trial {i}/{n_trials} "
                f"(cell probability threshold = {cellprob:.3f}, flow threshold = {flowthresh:.3f}, "
                f"niter = {niter}, minimum cell size = {min_size})"
            ),
        )

        masks_pred, _, _ = eval_model.eval(
            images,
            channels=channels,
            diameter=None,
            cellprob_threshold=cellprob,
            flow_threshold=flowthresh,
            niter=niter,
            min_size=min_size,
        )
        ap = metrics.average_precision(masks, masks_pred)[0]
        score = float(np.nanmean(ap[:, 0]))

        results.append(
            {
                "cellprob": cellprob,
                "flow_threshold": flowthresh,
                "niter": niter,
                "min_size": min_size,
                "ap_iou_0.5": score,
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(
        {"cellprob": 0.2, "flow_threshold": 0.4, "niter": 1000, "min_size": 100}
    )
    study.optimize(objective, n_trials=n_trials)
    pb.empty()

    df = pd.DataFrame(results).sort_values(
        by="ap_iou_0.5", ascending=False, na_position="last"
    )
    st.session_state["cp_grid_results_df"] = df

    set_cp_hparams(study.best_trial.params)

    if not df.empty and np.isfinite(df["ap_iou_0.5"].iloc[0]):
        best = df.iloc[0]
        set_cp_hparams(best)
        st.success(
            f"Best hyperparameters set: cellprob={best['cellprob']}, "
            f"flow={best['flow_threshold']}, min_size={int(best['min_size'])}, "
            f"niter={int(best['niter'])}"
        )
    else:
        st.info("No valid result to set best hyperparameters.")

    return channels


def validate_and_compare(images, masks, channels):
    with st.spinner(
        "Validating model... Not long to go! Don't click yet, you will interupt training."
    ):
        base_model = CellposeModel3Proxy(
            pretrained_model=ss["cp_base_model"], gpu=core.use_gpu()
        )

        hp = dict(
            diameter=None,
            cellprob_threshold=float(ss.get("cp_cellprob_threshold")),
            flow_threshold=float(ss.get("cp_flow_threshold")),
            niter=int(ss.get("cp_niter")),
            min_size=int(ss.get("cp_min_size")),
        )

        base_preds, _, _ = base_model.eval(images, channels=channels, **hp)
        tuned_model = get_cellpose_model()
        tuned_preds, _, _ = tuned_model.eval(images, channels=channels, **hp)

        base_ious = compute_prediction_ious(
            images=images, masks=masks, model=base_model, channels=channels
        )
        tuned_ious = compute_prediction_ious(
            images=images, masks=masks, model=tuned_model, channels=channels
        )
        ss["cellpose_iou_comparison"] = plot_iou_comparison(base_ious, tuned_ious)

        gt_counts = [int(np.count_nonzero(np.unique(mask))) for mask in masks]
        base_counts = [int(np.count_nonzero(np.unique(pred))) for pred in base_preds]
        tuned_counts = [int(np.count_nonzero(np.unique(pred))) for pred in tuned_preds]

        ss["cellpose_original_counts_comparison"] = plot_pred_vs_true_counts(
            gt_counts, base_counts, title="Base Model Predictions"
        )
        ss["cellpose_tuned_counts_comparison"] = plot_pred_vs_true_counts(
            gt_counts, tuned_counts, title="Tuned Model Predictions"
        )

        ss["cp_zip_bytes"] = build_cellpose_zip_bytes()


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

    go = st.button(
        "Fine-tune Cellpose",
        width="stretch",
        type="primary",
        disabled=other_job_running,
    )
    if not go:
        return

    # Start async training
    recs, base_model, epochs, lr, wd, nimg, channels, min_cells = get_train_setup()
    start_cellpose_training(
        recs, base_model, epochs, lr, wd, nimg, channels, min_train_masks=min_cells
    )
    st.rerun()


def show_cellpose_training_plots():
    """Render saved Cellpose plots from session state (if available)."""

    # check for previous fine-tuning data
    if "cellpose_training_losses" not in st.session_state:
        st.header("Cellpose Training Summary")
        st.info("No fine-tuning data to show.")
        return

    else:

        st.header("Cellpose Training Summary")
        cellpose_training_plot_help()

        col1, col2 = st.columns(2)
        with col1:

            # plot training losses
            st.plotly_chart(
                st.session_state["cellpose_training_losses"],
                width="stretch",
            )

            # plot original vs predicted counts
            if "cellpose_original_counts_comparison" in st.session_state:
                st.plotly_chart(
                    st.session_state["cellpose_original_counts_comparison"],
                    width="stretch",
                )

        with col2:

            # plot iou comparison
            if "cellpose_iou_comparison" in st.session_state:
                st.plotly_chart(
                    st.session_state["cellpose_iou_comparison"],
                    width="stretch",
                )

            # plot tuned vs predicted counts
            if "cellpose_tuned_counts_comparison" in st.session_state:
                st.plotly_chart(
                    st.session_state["cellpose_tuned_counts_comparison"],
                    width="stretch",
                )

        # display grid search results if applicable
        if ss.get("cp_do_gridsearch") and "cp_grid_results_df" in st.session_state:
            st.dataframe(
                st.session_state["cp_grid_results_df"],
                hide_index=True,
                width="stretch",
            )

        else:
            st.info("No hyperparameter tuning performed.")


@st.fragment(run_every=2)
def render_densenet_status_fragment():
    """Real-time status and log viewer for DenseNet training."""
    dn_job = st.session_state.get("dn_training_job")
    if not dn_job or dn_job.get("status") != "running":
        return

    status = check_densenet_training_status()
    ss.setdefault("dn_icon_toggle", True)
    icon = "⌛" if ss["dn_icon_toggle"] else "⏳"
    ss["dn_icon_toggle"] = not ss["dn_icon_toggle"]

    if status == "running":
        st.info(f"{icon} DenseNet training in progress...")

        log_content = dn_job.get("log_content", "")
        if log_content:
            with st.expander("📄 View Training Log", expanded=False):
                st.code(log_content, language="text")
        else:
            st.caption("Waiting for training output...")

        if st.button(
            "Cancel Training", type="primary", key="cancel_dn_frag", width="stretch"
        ):
            cancel_densenet_training()
            st.rerun()
    elif status == "complete":
        st.success("Finalizing DenseNet training...")
        st.balloons()  # cool as f*ck

        ss["dn_zip_bytes"] = build_densenet_zip_bytes(dn_job["input_size"])
        ss.pop("dn_training_job", None)
        st.rerun()
    elif status == "failed":
        st.error("❌ DenseNet training failed.")
        with st.expander("Show Error Details"):
            st.code(dn_job.get("error", "Unknown error"))
        # ss.pop("dn_training_job", None)
        # st.rerun()


@st.fragment(run_every=10)
def render_cellpose_status_fragment():
    """Real-time status and log viewer for Cellpose training and validation"""
    cp_job = st.session_state.get("cp_training_job")
    val_job = st.session_state.get("cp_validation_job")

    from src.helpers.cellpose_functions import (
        build_cellpose_zip_bytes,
    )

    ss.setdefault("cp_icon_toggle", True)
    icon = "⌛" if ss["cp_icon_toggle"] else "⏳"
    ss["cp_icon_toggle"] = not ss["cp_icon_toggle"]

    # Check Training
    if cp_job and cp_job.get("status") == "running":
        status = check_cellpose_training_status()

        if status == "running":
            st.info(f"{icon} Cellpose training in progress...")

            log_content = cp_job.get("log_content", "")
            if log_content:
                with st.expander("📄 View Training Log", expanded=False):
                    st.code(log_content, language="text")
            else:
                st.caption("Waiting for training output...")

            if st.button(
                "Cancel Training", type="primary", key="cancel_cp_frag", width="stretch"
            ):
                cancel_cellpose_training()
                st.rerun()
        elif status == "complete":
            st.success(
                "Finalizing Cellpose training..."
            )  # technically its done but the synchronous parts here are too slow

            from src.panels.fine_tune_panel import get_train_setup

            recs, base_model, epochs, lr, wd, nimg, channels, _ = get_train_setup()
            start_cellpose_validation(
                recs=recs,
                base_model=base_model,
                channels=channels,
                do_gridsearch=ss.get("cp_do_gridsearch", False),
                n_trials=ss.get("cp_n_trials", 20),
            )
            ss.pop("cp_training_job", None)
            st.rerun()
        elif status == "failed":
            st.error("❌ Cellpose training failed.")
            with st.expander("Show Error Details"):
                st.code(cp_job.get("error", "Unknown error"))
            # ss.pop("cp_training_job", None)
            # st.rerun()
        return

    # Check Validation
    if val_job and val_job.get("status") == "running":
        val_status = check_cellpose_validation_status()

        if val_status == "running":
            st.info(f"{icon} Cellpose validation in progress...")
            log_content = val_job.get("log_content", "")
            if log_content:
                with st.expander("📄 View Validation Log", expanded=False):
                    st.code(log_content, language="text")
            else:
                st.caption("Waiting for validation output...")

            if st.button(
                "Cancel Validation",
                type="primary",
                key="cancel_val_frag",
                width="stretch",
            ):
                cancel_cellpose_validation()
                st.rerun()
        elif val_status == "complete":
            st.success("Finalizing validation... please stay on this page for now!")
            st.balloons()
            ss["cp_zip_bytes"] = build_cellpose_zip_bytes()
            ss.pop("cp_validation_job", None)
            st.rerun()
        elif val_status == "failed":
            st.error("❌ Validation failed.")
            with st.expander("Show Error Details"):
                st.code(val_job.get("error", "Unknown error"))
            # ss.pop("cp_validation_job", None)
            # st.rerun()
