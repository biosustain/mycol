import os
import tempfile
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import cv2
from cellpose import core, metrics
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
    plot_loss_curve,
    snapshot_for_undo,
)
from src.helpers.job_runner import (
    start_worker_job,
    check_worker_job_status,
    cancel_worker_job,
)
from pathlib import Path
import plotly.graph_objects as go

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
    ss = st.session_state
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
    ss = st.session_state
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


def get_tuned_model():
    """
    Returns a CellposeModel3Proxy loaded with the fine-tuned weights
    --> ensures architecture consistency (training is cp3)
    """
    weights_path = get_cellpose_weights()
    if not weights_path:
        raise RuntimeError("No fine-tuned model weights found in session state.")

    return CellposeModel3Proxy(pretrained_model=weights_path, gpu=core.use_gpu())


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

    # reset diameter to None for automatic estimation
    # this is necessary be in the online version of the app only
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


class CellposeModel3Proxy:
    """A proxy class that runs Cellpose 3 inference via background worker"""

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
        from cellpose import models as cp_models, io

        is_list = isinstance(x, (list, tuple))
        images = x if is_list else [x]

        _ = io.logger_setup()
        cell_model = cp_models.CellposeModel(
            gpu=self.gpu, pretrained_model=self.pretrained_model
        )
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


def compute_prediction_ious(images, masks, model, channels):
    """
    Evaluate a Cellpose model on a list of images/masks and return the IoU values per image.

    Parameters
    ----------
    images : list of np.ndarray
        Input images to evaluate.
    masks : list of np.ndarray
        Corresponding ground-truth segmentation masks.
    model : cellpose.models.CellposeModel
        A loaded Cellpose model (base or fine-tuned).

    Returns
    -------
    list of float
        IoU value for each image.
    """
    preds, _, _ = model.eval(images, channels=channels)
    ious = [
        metrics.average_precision([gt], [pr])[0][:, 0].mean()
        for gt, pr in zip(masks, preds)
    ]
    return ious


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

    # create figure
    fig = go.Figure(layout=dict(barcornerradius=10))
    fig.add_bar(
        x=x,
        y=means,
        width=0.6,
        error_y=dict(type="data", array=sds, visible=True),
        marker=dict(
            color=[
                "#EBF1F8",
                "#EBF1F8",
            ],
            line=dict(color="#004280", width=2),
        ),
    )

    # add individual data points with jitter
    j = 0.12
    fig.add_scatter(
        x=(np.full(len(base_ious), x[0]) + (np.random.rand(len(base_ious)) - 0.5) * j),
        y=base_ious,
        mode="markers",
        marker=dict(color="#004280", size=6),
        text=names,
        hovertemplate="%{text}<br>Mean IoU: %{y:.3f}<extra></extra>",
    )
    fig.add_scatter(
        x=(
            np.full(len(tuned_ious), x[1]) + (np.random.rand(len(tuned_ious)) - 0.5) * j
        ),
        y=tuned_ious,
        mode="markers",
        marker=dict(color="#004280", size=6),
        text=names,
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

    # create figure
    fig = go.Figure()
    fig.add_scatter(
        x=gt_counts,
        y=base_counts,
        mode="markers",
        marker=dict(size=8, opacity=0.85, color="#004280"),
        name="Original",
        text=names,
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


@st.cache_resource
def load_base_cellpose_model(base_model: str):
    """Loads a base Cellpose model for fine-tuning."""
    init_model = None if base_model == "scratch" else base_model
    return CellposeModel3Proxy(pretrained_model=init_model, gpu=core.use_gpu())


def start_cellpose_training(
    recs: dict,
    base_model: str,
    epochs=100,
    learning_rate=0.1,
    weight_decay=0.0001,
    nimg_per_epoch=32,
    channels=[0, 0],
    min_train_masks=5,
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
            nimg_per_epoch=nimg_per_epoch,
            channels=np.array(channels),
            min_train_masks=min_train_masks,
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
        ss = st.session_state
        ss["cellpose_model_bytes"] = buf.getvalue()
        ss["cellpose_model_name"] = model_name
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
    recs, base_model, channels, do_gridsearch=False, n_trials=20
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
        ),
    )


def check_cellpose_validation_status():
    """Check validation status and load results when complete."""

    def _on_complete(data, job):
        ss = st.session_state
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


def build_cellpose_zip_bytes():
    """Builds a zip file containing the fine-tuned Cellpose model, training parameters,
    images, masks, and plots. Returns the zip file as bytes."""

    ok = ordered_keys()
    ss = st.session_state
    n_masks = sum(
        int(
            len(np.unique(ss["images"][k].get("masks", np.array([], dtype=np.uint16))))
            - 1
        )
        for k in ok
    )

    # extract training parameters; an uploaded (vs trained) model has none of
    # these set, so coerce only when present and skip missing entries
    def _coerce(key, cast):
        val = ss.get(key)
        return cast(val) if val is not None else None

    params = {
        k: v
        for k, v in dict(
            base_model=ss.get("cp_base_model"),
            epochs=_coerce("cp_max_epoch", int),
            learning_rate=_coerce("cp_learning_rate", float),
            weight_decay=_coerce("cp_weight_decay", float),
            batch_size=_coerce("cp_batch_size", int),
            images_used=len(ok),
            masks_used=n_masks,
        ).items()
        if v is not None
    }

    # Build zip in memory
    buf = IO.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])

        # Optional: grid search results
        if ss.get("cp_do_gridsearch") and ss.get("cp_grid_results_df") is not None:
            z.writestr(
                "hyperparameter_search.csv",
                ss["cp_grid_results_df"].to_csv(index=False),
            )

        # Training parameters table
        z.writestr(
            "params.csv",
            pd.Series(params)
            .rename_axis("parameter")
            .reset_index(name="value")
            .to_csv(index=False),
        )

        # Images and masks
        for k in ok:
            rec = ss["images"][k]
            img_name = Path(rec.get("name")).stem

            b = IO.BytesIO()
            Image.fromarray(np.asarray(rec["image"])).save(b, "TIFF")
            z.writestr(f"images/{img_name}.tif", b.getvalue())

            c = IO.BytesIO()
            Image.fromarray(np.asarray(rec["masks"])).save(c, "TIFF")
            z.writestr(f"masks/{img_name}_masks.tif", c.getvalue())

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
        st.session_state["edit_canvas_nonce"] += 1
    st.rerun()


def batch_segment_and_refresh(model_type: str | None = None):
    """calls cellpose to segment all images with progress bar"""
    ok = ordered_keys()
    params = get_cellpose_hparams_from_state()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        segment_with_cellpose(
            st.session_state.images.get(k), model_type=model_type, **params
        )
        pb.progress(i / n, text=f"Segmented {i}/{n}")


def get_cellpose_hparams_from_state():
    """calls hparam values from session state"""
    # Build kwargs matching segment_rec_with_cellpose signature
    ch1 = int(st.session_state.get("cp_ch1"))
    ch2 = int(st.session_state.get("cp_ch2"))
    diameter = st.session_state.get("cp_diameter")

    return dict(
        channels=(ch1, ch2),
        diameter=diameter,
        cellprob_threshold=float(st.session_state.get("cp_cellprob_threshold")),
        flow_threshold=float(st.session_state.get("cp_flow_threshold")),
        min_size=int(st.session_state.get("cp_min_size")),
        niter=int(st.session_state.get("cp_niter")),
    )
