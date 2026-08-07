import os
import tempfile
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import cv2
from cellpose import core
import torch
from PIL import Image
import io as IO
from sklearn.metrics import r2_score, mean_absolute_error
import zipfile
from src.helpers.state_ops import (
    ordered_keys,
    get_current_rec,
    normalize_image,
    add_plotly_as_png_to_zip,
    params_to_csv,
    write_image_to_zip,
    write_mask_to_zip,
    snapshot_for_undo,
    image_number_lookup,
)
from src.helpers.plot_helpers import (
    plot_loss_curve,
    point_hover_texts,
    NAVY,
    PALE_BLUE,
)
from src.helpers.job_runner import (
    start_worker_job,
    check_worker_job_status,
    cancel_worker_job,
)
from pathlib import Path
import plotly.graph_objects as go

ss = st.session_state

# -----------------------------------------------------#
# ---------------- IMAGE PREPROCESSING --------------- #
# -----------------------------------------------------#


def preprocess_for_cellpose(rec):
    """takes record input and prepares the stored image for cellpose"""

    img = rec["image"]

    # convert to grayscale if needed
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {img.shape}; expected (H,W) or (H,W,C)"
        )

    # normalize
    im_in = normalize_image(img)

    return im_in


def convert_cellpose_mask_to_single_array(mask_output, H, W):
    """Converts Cellpose output mask to single (H,W) label image with contiguous ids 1..N"""

    # handle empty mask case
    if mask_output is None or mask_output.size == 0:
        inst = np.zeros((H, W), dtype=np.uint8)
        return inst
    # handle standard case
    else:
        a = np.asarray(mask_output)
        if a.shape != (H, W):
            # (rare) ensure correct size; nearest preserves labels
            a = np.array(
                Image.fromarray(a).resize((W, H), Image.NEAREST), dtype=a.dtype
            )
        # remap ids to contiguous 1..K
        vals = np.unique(a)
        ids = vals[vals > 0]
        if ids.size == 0:
            inst = np.zeros((H, W), dtype=np.uint8)
            K = 0
        else:
            # remap old ids -> 1..K (contiguous)
            K = int(ids.size)
            max_old = int(a.max())
            lut_dtype = np.uint32 if K > np.iinfo(np.uint16).max else np.uint16
            lut = np.zeros(max_old + 1, dtype=lut_dtype)
            lut[ids] = np.arange(1, K + 1, dtype=lut_dtype)
            inst = lut[a]

        return inst


# -----------------------------------------------------#
# ---------------- CELLPOSE INFERENCE ---------------- #
# -----------------------------------------------------#


# --- materialize session model bytes to a stable temp path ---
def get_cellpose_weights() -> str | None:
    """writes Cellpose model bytes from session state to a temp file and returns the path"""
    b = ss.get("cellpose_model_bytes", None)
    name = ss.get("cellpose_model_name", None)
    if not b or not name:
        return None

    h = hashlib.sha1(b).hexdigest()[:12]
    suffix = os.path.splitext(name)[1] or ".npy"
    path = os.path.join(tempfile.gettempdir(), f"cellpose_{h}{suffix}")

    # write once; if the file exists, assume it matches the hash
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(b)
    return path


def get_cellpose_model():
    tag = (
        hashlib.sha1(ss["cellpose_model_bytes"]).hexdigest()[:12]
        if ss.get("cellpose_model_bytes")
        else "cyto2"
    )

    if ss.get("cellpose_model_obj") is not None and ss.get("cellpose_model_tag") == tag:
        return ss["cellpose_model_obj"]

    weights_path = get_cellpose_weights()
    model_type = "cyto2"
    if weights_path:
        model_type = weights_path

    model = CellposeModel3Proxy(pretrained_model=model_type, gpu=core.use_gpu())

    ss["cellpose_model_obj"] = model
    ss["cellpose_model_tag"] = tag

    return model


def segment_with_cellpose(
    rec: dict,
    *,
    model_type: str | None = None,
    channels=(0, 0),
    diameter=None,
    cellprob_threshold=-0.2,
    flow_threshold=0.4,
    min_size=0,
    niter=0,
) -> dict:
    """
    Runs Cellpose on rec['image'] and overwrites rec['masks'] with a single (H,W)
    integer label image (0=background, 1..N=instances). Resets rec['labels'].
    If model_type is given (e.g. "cyto2", "cyto3"), that base model is used directly.
    """

    im_in = preprocess_for_cellpose(rec)

    if model_type is not None:
        cell_model = CellposeModel3Proxy(
            pretrained_model=model_type, gpu=core.use_gpu()
        )
    else:
        cell_model = get_cellpose_model()

    # the UI uses 0 to mean "estimate automatically"; Cellpose expects None
    if diameter == 0:
        diameter = None

    masks_out, flows, styles = cell_model.eval(
        [im_in],
        channels=list(channels),
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=min_size,
        niter=niter,
    )
    mask_output = masks_out[0] if isinstance(masks_out, (list, tuple)) else masks_out

    # set record masks to new predicted mask matrix
    rec["masks"] = convert_cellpose_mask_to_single_array(
        mask_output, rec["H"], rec["W"]
    )
    # clear any labels in the record (no new masks are labelled)
    rec["labels"] = {
        int(i): None for i in np.unique(rec["masks"]) if i != 0
    }  # reset/realign


@st.cache_resource(show_spinner="Loading Cellpose model…")
def _load_cellpose_model(pretrained_model: str, gpu: bool):
    """Load a Cellpose model once and reuse it across reruns"""
    from cellpose import models as cp_models, io

    _ = io.logger_setup()
    return cp_models.CellposeModel(gpu=gpu, pretrained_model=pretrained_model)


class CellposeModel3Proxy:
    """Thin in-process wrapper around a (cached) Cellpose 3 model."""

    def __init__(self, pretrained_model, gpu=True):
        self.pretrained_model = pretrained_model
        self.gpu = gpu
        # attributes for API compatibility
        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.net = type("obj", (object,), {"device": self.device})()

    def eval(
        self,
        x,
        channels=None,
        diameter=None,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        min_size=15,
        niter=200,
        **kwargs,
    ):
        is_list = isinstance(x, (list, tuple))
        images = x if is_list else [x]

        cell_model = _load_cellpose_model(self.pretrained_model, self.gpu)
        masks, flows, styles = cell_model.eval(
            images,
            channels=channels if channels is not None else [0, 0],
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            niter=niter,
        )

        return (masks, flows, styles) if is_list else (masks[0], flows, styles)


# -----------------------------------------------------#
# ----------------- CELLPOSE FIGURES ----------------- #
# -----------------------------------------------------#


def plot_iou_comparison(base_ious, tuned_ious, image_names=None):
    """Plots a bar chart comparing mean IoU of base and fine-tuned Cellpose models with error bars."""

    # prepare data
    labels = ["Base Model", "Fine-tuned"]
    x = [0, 1]
    means = [np.mean(base_ious), np.mean(tuned_ious)]
    sds = [
        np.std(base_ious, ddof=1) if len(base_ious) > 1 else 0.0,
        np.std(tuned_ious, ddof=1) if len(tuned_ious) > 1 else 0.0,
    ]
    names = (
        image_names if image_names else [f"Image {i}" for i in range(len(base_ious))]
    )
    lookup = image_number_lookup()
    hover = point_hover_texts([lookup.get(n, "?") for n in names], names)

    # create figure
    fig = go.Figure(layout=dict(barcornerradius=10))
    fig.add_bar(
        x=x,
        y=means,
        width=0.6,
        error_y=dict(type="data", array=sds, visible=True),
        marker=dict(
            color=[PALE_BLUE, PALE_BLUE],
            line=dict(color=NAVY, width=2),
        ),
    )

    # add individual data points with jitter
    j = 0.12
    fig.add_scatter(
        x=(np.full(len(base_ious), x[0]) + (np.random.rand(len(base_ious)) - 0.5) * j),
        y=base_ious,
        mode="markers",
        marker=dict(color=NAVY, size=6),
        text=hover,
        hovertemplate="%{text}<br>Mean IoU: %{y:.3f}<extra></extra>",
    )
    fig.add_scatter(
        x=(
            np.full(len(tuned_ious), x[1]) + (np.random.rand(len(tuned_ious)) - 0.5) * j
        ),
        y=tuned_ious,
        mode="markers",
        marker=dict(color=NAVY, size=6),
        text=hover,
        hovertemplate="%{text}<br>Mean IoU: %{y:.3f}<extra></extra>",
    )

    # layout settings
    fig.update_layout(
        title="IoU Comparison",
        xaxis=dict(tickmode="array", tickvals=x, ticktext=labels, range=[-0.6, 1.6]),
        yaxis=dict(title="Mean IoU", range=[0, 1.05]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=450,
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.1)")

    return fig


def plot_pred_vs_true_counts(gt_counts, base_counts, title, image_names=None):
    """Plots predicted vs true counts scatter plot with R² and MAE annotations."""

    # determine plot limits
    lim = max(1, max(gt_counts + base_counts))
    names = (
        image_names if image_names else [f"Image {i}" for i in range(len(gt_counts))]
    )
    lookup = image_number_lookup()
    hover = point_hover_texts([lookup.get(n, "?") for n in names], names)

    # create figure
    fig = go.Figure()
    fig.add_scatter(
        x=gt_counts,
        y=base_counts,
        mode="markers",
        marker=dict(size=8, opacity=0.85, color=NAVY),
        name="Original",
        text=hover,
        hovertemplate="%{text}<br>True: %{x}<br>Predicted: %{y}<extra></extra>",
    )
    fig.add_scatter(
        x=[0, lim],
        y=[0, lim],
        mode="lines",
        line=dict(dash="dash", width=1, color="gray"),
        showlegend=False,
    )
    # add annotations if more than one data point
    if len(gt_counts) > 1:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.95,
            showarrow=False,
            text=f"R² = {r2_score(gt_counts, base_counts):.3f}<br>MAE = {mean_absolute_error(gt_counts, base_counts):.3f}",
            bgcolor="white",
            opacity=0.7,
            align="left",
        )

    # layout settings
    fig.update_layout(
        title=title,
        xaxis_title="True number of masks",
        yaxis_title="Predicted number of masks",
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        height=450,
    )

    # set axes ranges and grid
    fig.update_xaxes(range=[-0.5, lim + 0.5], showgrid=True)
    fig.update_yaxes(
        range=[-0.5, lim + 0.5], showgrid=True, gridcolor="rgba(0,0,0,0.1)"
    )
    return fig


# -----------------------------------------------------#
# ---------------- FINE TUNE CELLPOSE ---------------- #
# -----------------------------------------------------#


def start_cellpose_training(
    recs: dict,
    base_model: str,
    epochs=100,
    learning_rate=0.1,
    weight_decay=0.0001,
    batch_size=8,
    nimg_per_epoch=None,
    channels=[0, 0],
    min_train_masks=5,
    test_split=0.2,
):
    """Starts Cellpose fine-tuning asynchronously using cp3 worker bridge"""
    images, masks = [], []
    for k in recs:
        images.append(preprocess_for_cellpose(recs[k]))
        masks.append(recs[k]["masks"].astype("uint16"))

    start_worker_job(
        job_key="cp_training_job",
        worker_name="finetune",
        inputs=dict(
            images=np.array([np.ascontiguousarray(im) for im in images], dtype=object),
            masks=np.array(
                [np.ascontiguousarray(m).astype(np.uint16) for m in masks], dtype=object
            ),
            base_model=base_model,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            nimg_per_epoch=(nimg_per_epoch or 0),  # 0 = None sentinel (all images)
            channels=np.array(channels),
            min_train_masks=min_train_masks,
            test_split=test_split,
        ),
        metadata={"base_model": base_model, "num_images": len(images)},
    )


def check_cellpose_training_status():
    """Check if Cellpose training is complete and load results if so."""

    def _on_complete(data, job):
        train_losses = np.array(data["train_losses"])
        test_losses = np.array(data["test_losses"])
        model_name = str(data["model_name"])
        state_dict = data["state_dict"].item()

        buf = IO.BytesIO()
        torch.save(state_dict, buf)
        ss["cellpose_model_bytes"] = buf.getvalue()
        ss["cellpose_model_name"] = model_name

        # drop the previous model so the newly trained weights are loaded
        _load_cellpose_model.clear()
        ss["model_to_fine_tune"] = job["base_model"]
        ss["train_losses"] = train_losses
        ss["test_losses"] = test_losses
        ss["cellpose_training_losses"] = plot_loss_curve(train_losses, test_losses)

        job["train_losses"] = train_losses
        job["test_losses"] = test_losses
        job["model_name"] = model_name

    return check_worker_job_status("cp_training_job", _on_complete)


def cancel_cellpose_training():
    cancel_worker_job("cp_training_job")


def start_cellpose_validation(
    recs, base_model, channels, do_gridsearch=False, n_trials=20, test_split=0.2
):
    model_path = get_cellpose_weights()
    if not model_path:
        st.error("No trained model found")
        return

    from src.panels.fine_tune_panel import prepare_eval_data  # TODO: move up maybe?

    images, masks, image_names = prepare_eval_data(recs)

    start_worker_job(
        job_key="cp_validation_job",
        worker_name="validation",
        inputs=dict(
            images=np.array(images, dtype=object),
            masks=np.array(masks, dtype=object),
            image_names=np.array(image_names, dtype=object),
            base_model=base_model,
            tuned_model_path=model_path,
            channels=np.array(channels),
            do_gridsearch=do_gridsearch,
            n_trials=n_trials,
            test_split=test_split,
        ),
    )


def check_cellpose_validation_status():
    """Check validation status and load results when complete."""

    def _on_complete(data, job):
        optuna_results = data.get("optuna_results")
        best_params = data["best_params"].item()
        validation_metrics = data["validation_metrics"].item()

        try:
            if optuna_results is not None:
                if isinstance(optuna_results, np.ndarray):
                    optuna_results = (
                        optuna_results.item()
                        if optuna_results.ndim == 0
                        else optuna_results.tolist()
                    )

                df = pd.DataFrame(optuna_results)
                if not df.empty and "ap_iou_0.5" in df.columns:
                    ss["cp_grid_results_df"] = df.sort_values(
                        by="ap_iou_0.5", ascending=False, na_position="last"
                    )
                else:
                    ss["cp_grid_results_df"] = df
                    if not df.empty:
                        print(
                            "Warning: 'ap_iou_0.5' column missing from Optuna results. Skipping sort."
                        )

                if best_params:
                    ss["cp_cellprob_threshold"] = float(
                        best_params.get("cellprob", 0.0)
                    )
                    ss["cp_flow_threshold"] = float(
                        best_params.get("flow_threshold", 0.4)
                    )
                    ss["cp_min_size"] = int(best_params.get("min_size", 15))
                    ss["cp_niter"] = int(best_params.get("niter", 200))
        except Exception as e:
            st.error(f"Error processing Optuna results: {e}")

        image_names = validation_metrics.get("image_names", [])
        ss["cellpose_iou_comparison"] = plot_iou_comparison(
            validation_metrics["base_ious"],
            validation_metrics["tuned_ious"],
            image_names=image_names,
        )
        ss["cellpose_original_counts_comparison"] = plot_pred_vs_true_counts(
            validation_metrics["gt_counts"],
            validation_metrics["base_counts"],
            title="Base Model Predictions",
            image_names=image_names,
        )
        ss["cellpose_tuned_counts_comparison"] = plot_pred_vs_true_counts(
            validation_metrics["gt_counts"],
            validation_metrics["tuned_counts"],
            title="Tuned Model Predictions",
            image_names=image_names,
        )

        ss["cp_zip_bytes"] = build_cellpose_zip_bytes()

    return check_worker_job_status("cp_validation_job", _on_complete)


def cancel_cellpose_validation():
    cancel_worker_job("cp_validation_job")


def is_not_empty_mask(m):
    """returns True if mask is a non-empty numpy array"""
    return isinstance(m, np.ndarray) and m.any()


def write_cellpose_param_csvs(zf) -> None:
    """Write the Cellpose training, inference and grid-search CSVs into `zf`."""
    # images are converted to grayscale before Cellpose, so channels are fixed at 0
    training = {
        "base_model": ss.get("cp_base_model"),
        "max_epoch": ss.get("cp_max_epoch"),
        "learning_rate": ss.get("cp_learning_rate"),
        "weight_decay": ss.get("cp_weight_decay"),
        "batch_size": ss.get("cp_batch_size"),
        "nimg_per_epoch": ss.get("cp_nimg_per_epoch"),
        "min_cells_per_image": ss.get("cp_min_cells_per_image"),
        "test_split": ss.get("cp_test_split", 0.2),
        "training_ch1": 0,
        "training_ch2": 0,
        "do_gridsearch": ss.get("cp_do_gridsearch", False),
        "n_trials": ss.get("cp_n_trials", 20),
    }
    inference = {
        "channel_1": 0,
        "channel_2": 0,
        "diameter": ss.get("cp_diameter"),
        "cellprob_threshold": ss.get("cp_cellprob_threshold"),
        "flow_threshold": ss.get("cp_flow_threshold"),
        "min_size": ss.get("cp_min_size"),
        "niter": ss.get("cp_niter"),
    }
    zf.writestr("cellpose_training_hyperparameters.csv", params_to_csv(training))
    zf.writestr("cellpose_inference_hyperparameters.csv", params_to_csv(inference))
    grid = ss.get("cp_grid_results_df")
    if grid is not None:
        zf.writestr("cellpose_grid_search_results.csv", grid.to_csv(index=False))


def build_cellpose_zip_bytes():
    """Build a zip with the fine-tuned Cellpose model, parameters, images, masks,
    and plots. Returns the zip as bytes."""

    ok = ordered_keys()

    buf = IO.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])
        write_cellpose_param_csvs(z)

        # Images and masks
        mask_suffix = ss.get("mask_suffix", "_masks")
        for k in ok:
            rec = ss["images"][k]
            img_name = Path(rec.get("name")).stem
            write_image_to_zip(z, img_name, rec["image"])
            write_mask_to_zip(z, img_name, rec["masks"], mask_suffix)

        # Plots
        add_plotly_as_png_to_zip(
            "cellpose_training_losses", z, "plots/cellpose_training_losses.png"
        )
        add_plotly_as_png_to_zip(
            "cellpose_iou_comparison", z, "plots/cellpose_iou_comparison.png"
        )
        add_plotly_as_png_to_zip(
            "cellpose_original_counts_comparison",
            z,
            "plots/cellpose_original_counts_comparison.png",
        )
        add_plotly_as_png_to_zip(
            "cellpose_tuned_counts_comparison",
            z,
            "plots/cellpose_tuned_counts_comparison.png",
        )

    return buf.getvalue()


# -----------------------------------------------------#
# ----------     SEGMENTATION FUNCTIONS.     --------- #
# -----------------------------------------------------#


def segment_current_and_refresh(model_type: str | None = None):
    """calls cellpose to segment the current image"""
    rec = get_current_rec()
    if rec is not None:
        params = get_cellpose_hparams_from_state()
        # snapshot so this single-image segmentation can be undone (batch path doesn't)
        snapshot_for_undo(rec)
        segment_with_cellpose(rec, model_type=model_type, **params)
    st.rerun()


def batch_segment_and_refresh(model_type: str | None = None):
    """calls cellpose to segment all images with progress bar"""
    ok = ordered_keys()
    params = get_cellpose_hparams_from_state()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_with_cellpose(
            ss.images.get(k), model_type=model_type, **params
        )
        pb.progress(i / n, text=f"Segmented {i}/{n}")


def get_cellpose_hparams_from_state():
    """calls hparam values from session state"""
    # Build kwargs matching segment_rec_with_cellpose signature
    # images are converted to grayscale in preprocess_for_cellpose, so channels are fixed
    diameter = ss.get("cp_diameter")

    return dict(
        channels=(0, 0),
        diameter=diameter,
        cellprob_threshold=float(ss.get("cp_cellprob_threshold")),
        flow_threshold=float(ss.get("cp_flow_threshold")),
        min_size=int(ss.get("cp_min_size")),
        niter=int(ss.get("cp_niter")),
    )
