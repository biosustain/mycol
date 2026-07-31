import copy
import io
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import tifffile as tiff
import plotly.io as pio

ss = st.session_state

# Fixed by the Cellpose-SAM paper's fine-tuning protocol, not user-editable: AdamW at
# lr 1e-5 with weight decay 0.1. Cellpose 3's CNN-scale lr diverges on a transformer.
# Read these directly rather than from session state, which survives code reloads and
# is repopulated when a session saved by an older build is loaded.
CP_LEARNING_RATE = 1e-5
CP_WEIGHT_DECAY = 0.1

# Plotly figures persisted in the session zip as "{key}.json".
SESSION_PLOT_KEYS = [
    "cellpose_training_losses",
    "cellpose_iou_comparison",
    "cellpose_original_counts_comparison",
    "cellpose_tuned_counts_comparison",
    "densenet_training_losses",
    "densenet_training_metrics",
    "densenet_confusion_matrix",
]


_DEFAULTS = {
    # app-level state
    "images": {},                       # {order_key:int -> record:dict}
    "name_to_key": {},                  # {filename:str -> order_key:int}
    "current_key": None,                # active order_key
    "next_ord": 1,                      # next order_key to assign
    "cellpose_model_bytes": None,
    "cellpose_model_name": None,
    "densenet_ckpt_bytes": None,
    "densenet_ckpt_name": None,
    "side_new_label": "",
    "show_overlay": True,
    "show_normalized": True,
    "interaction_mode": "Remove",
    "skipped_files": [],
    "remove_click": False,
    "class_click": False,
    "last_class_xy": None,
    "last_remove_xy": None,
    "zoom": 1.0,

    # cellpose model training
    "cyto_to_train": "cpsam_v2",
    "cp_learning_rate": CP_LEARNING_RATE,
    "cp_weight_decay": CP_WEIGHT_DECAY,
    "train_losses": [],
    "test_losses": [],
    # cellpose inference
    "cp_min_size": 0,
    "cp_niter": 200,
    "cp_flow_threshold": 0.3,
    "cp_cellprob_threshold": 0.2,
    "cp_diameter": 0,

    # densenet training
    "dn_input_size": 64,
    "dn_batch_size": 32,
    "dn_max_epoch": 100,
    "dn_val_split": 0.2,

    # densenet model
    "densenet_model": None,

    # image dataset download options
    "dl_normalize_download": False,

    # class defaults
    "all_classes": ["No label"],
    "side_current_class": "No label",
    "cp_grid_results_df": None,
    "densenet_class_map": {},           # {pred_class_idx:int -> app_label:str}
}


def reset_global_state_defaults() -> None:
    """Ensure every session-state key used across panels has its default value.

    Idempotent: existing values are preserved, missing ones get a fresh copy of
    their default. Safe to call on every Streamlit rerun.
    """
    for k, v in _DEFAULTS.items():
        ss.setdefault(k, copy.deepcopy(v))


def stem(p: str) -> str:
    return Path(p).stem


# ss["view"] = (ox, oy, s): the annotate display shows a zoomed crop starting at
# full-res pixel (ox, oy), with s full-res pixels per display pixel. At zoom 1 this
# is (0, 0, W/disp_w), so these reduce to the original whole-image mapping.
def disp_to_full(x, y):
    """Map a display (viewport) coordinate to full-resolution image coordinates."""
    ox, oy, s = ss.get("view", (0.0, 0.0, 1.0))
    return ox + x * s, oy + y * s


def full_to_disp(x, y):
    """Map a full-resolution image coordinate to display (viewport) coordinates."""
    ox, oy, s = ss.get("view", (0.0, 0.0, 1.0))
    return (x - ox) / s, (y - oy) / s


def ordered_keys():
    return sorted(ss.images.keys())


def image_number_lookup():
    """Map each image name to its 1-based 'No.' in the image selection table."""
    return {
        ss["images"][k]["name"]: i
        for i, k in enumerate(ordered_keys(), start=1)
    }


def selected_training_keys(namespace: str):
    """Ordered image keys chosen for training in ``namespace`` (``'cp'``/``'dn'``).

    Falls back to every image when no explicit selection has been made yet.
    """
    selected = ss.get(f"{namespace}_selected_image_keys")
    if selected is None:
        return ordered_keys()
    selected = set(selected)
    return [k for k in ordered_keys() if k in selected]


def require_images():
    """Stop the page with a warning if no images have been uploaded yet."""
    if not ss["images"]:
        st.markdown(
            """<div style="background:rgba(255,135,0,0.12);border-left:4px solid #ff8700;border-radius:0 8px 8px 0;padding:14px 18px;">
            <p style="margin:0;font-size:1rem;color:#ff8700;font-weight:600;letter-spacing:0.05em;">NO IMAGES UPLOADED</p>
            <p style="margin:4px 0 0;font-size:1.5rem;font-weight:700;">Upload data first</p>
            <p style="margin:2px 0 0;font-size:1.1rem;opacity:0.75;">Please upload images on the 'Upload Models and Data' tab.</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.stop()


def get_current_rec():
    k = ss.get("current_key")
    return ss.images.get(k) if k is not None else None


def snapshot_for_undo(rec) -> None:
    """Save the one app-wide undo snapshot for the current image.

    Call immediately before any action that mutates the current record."""
    if rec is None:
        return
    masks = rec.get("masks")
    ss["undo"] = {
        "key": ss.get("current_key"),
        "masks": None if masks is None else masks.copy(),
        "labels": dict(rec.get("labels", {})),
        "boxes": list(rec.get("boxes", [])),
    }


def apply_undo(rec) -> bool:
    """Restore the undo snapshot if it's for the current image; always consume it."""
    snap = ss.pop("undo", None)
    if not snap or rec is None:
        return False
    if snap.get("key") != ss.get("current_key"):
        return False  # snapshot is for a different image
    rec["masks"] = snap["masks"]
    rec["labels"] = snap["labels"]
    rec["boxes"] = snap["boxes"]
    return True


def reset_undo_on_navigation() -> None:
    """Drop the undo snapshot if the displayed image changed since it was taken."""
    snap = ss.get("undo")
    if snap and snap.get("key") != ss.get("current_key"):
        ss.pop("undo", None)


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    ss.current_key = ok[idx % len(ok)]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image intensities to uint8, for Cellpose, DenseNet patches and downloads.
    Scales mean intensity to ~127.5, or to the full uint8 range if mean <= 0.
    """
    im = image.astype(np.float32)
    if im.size == 0:
        return im

    mean_val = float(im.mean())
    if mean_val <= 0:
        # fallback: scale to full uint8 range
        rng = float(im.max() - im.min())
        im = (im - im.min()) / rng * 255.0 if rng > 0 else im * 0.0
    else:
        # scale by ratio so mean intensity ≈ 127.5 (mid-gray)
        im = im * (127.5 / mean_val)

    # ensure valid uint8 range
    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)


def params_to_csv(params: dict) -> str:
    """Serialize a {name: value} dict to a two-column parameter/value CSV string."""
    return (
        pd.Series(params)
        .rename_axis("parameter")
        .reset_index(name="value")
        .to_csv(index=False)
    )


def write_image_to_zip(zip_file, name, image) -> None:
    """Write an RGB image array to `zip_file` as images/{name}.tif (deflate-compressed)."""
    buf = io.BytesIO()
    tiff.imwrite(buf, np.asarray(image), photometric="rgb", compression="deflate")
    zip_file.writestr(f"images/{name}.tif", buf.getvalue())


def write_mask_to_zip(zip_file, name, mask, mask_suffix) -> None:
    """Write a label mask array to `zip_file` as masks/{name}{mask_suffix}.tif (uint16)."""
    buf = io.BytesIO()
    tiff.imwrite(buf, mask.astype(np.uint16))
    zip_file.writestr(f"masks/{name}{mask_suffix}.tif", buf.getvalue())


def add_plotly_as_png_to_zip(fig_key, zip_file, out_path, default_w=900, default_h=400):
    """Adds a plotly figure stored in st.session_state[fig_key] as a PNG to the given zip file.

    Silently skips figures that were never created (e.g. when a model was uploaded
    rather than trained, so no training plots exist)."""
    fig = ss.get(fig_key)
    if fig is None:
        return
    png = pio.to_image(
        fig,
        format="png",
        scale=3,
        width=int(getattr(fig.layout, "width", default_w) or default_w),
        height=int(getattr(fig.layout, "height", default_h) or default_h),
    )
    zip_file.writestr(out_path, png)
