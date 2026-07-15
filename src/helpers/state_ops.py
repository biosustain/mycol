import copy
from pathlib import Path
import streamlit as st
import numpy as np
import plotly.io as pio
import plotly.graph_objects as go


_DEFAULTS = {
    # app-level state
    "images": {},                       # {order_key:int -> record:dict}
    "name_to_key": {},                  # {filename:str -> order_key:int}
    "current_key": None,                # active order_key
    "next_ord": 1,                      # next order_key to assign
    "analysis_plots": [],
    "cellpose_model_bytes": None,
    "cellpose_model_name": None,
    "densenet_ckpt_bytes": None,
    "densenet_ckpt_name": None,
    "side_new_label": "",
    "show_overlay": True,
    "show_normalized": True,
    "interaction_mode": "Remove",
    "side_interaction_mode": "Draw box",
    "skipped_files": [],
    "remove_click": False,
    "class_click": False,
    "last_class_xy": None,
    "last_remove_xy": None,
    "disp_w": 0,

    # cellpose model training
    "cyto_to_train": "cyto3",
    "train_losses": [],
    "test_losses": [],
    # cellpose inference
    "cp_min_size": 0,
    "cp_niter": 500,
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

    # UI defaults / nonces
    "pred_canvas_nonce": 0,
    "edit_canvas_nonce": 0,
    "mask_uploader_nonce": 0,
    "image_uploader_nonce": 0,
    "side_panel": "Upload data",

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
    ss = st.session_state
    for k, v in _DEFAULTS.items():
        ss.setdefault(k, copy.deepcopy(v))


def stem(p: str) -> str:
    return Path(p).stem


def ordered_keys():
    return sorted(st.session_state.images.keys())


def image_number_lookup():
    """Map each image name to its 1-based 'No.' in the image selection table."""
    return {
        st.session_state["images"][k]["name"]: i
        for i, k in enumerate(ordered_keys(), start=1)
    }


def point_hover_texts(numbers, names, patches=None):
    """Per-point hover labels — image number, image name and (optionally) patch
    number, each on its own line. Shared by the cell-metrics and fine-tuning plots."""
    if patches is None:
        patches = [None] * len(names)
    texts = []
    for num, name, patch in zip(numbers, names, patches):
        line = f"Image number: {num}<br>Image name: {name}"
        if patch is not None:
            line += f"<br>Patch number: {patch}"
        texts.append(line)
    return texts


def selected_training_keys(namespace: str):
    """Ordered image keys chosen for training in ``namespace`` (``'cp'``/``'dn'``).

    Falls back to every image when no explicit selection has been made yet.
    """
    selected = st.session_state.get(f"{namespace}_selected_image_keys")
    if selected is None:
        return ordered_keys()
    selected = set(selected)
    return [k for k in ordered_keys() if k in selected]


def require_images():
    """Stop the page with a warning if no images have been uploaded yet."""
    if not st.session_state["images"]:
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
    k = st.session_state.get("current_key")
    return st.session_state.images.get(k) if k is not None else None


def snapshot_for_undo(rec) -> None:
    """Save the one app-wide undo snapshot for the current image.

    Call immediately before any action that mutates the current record."""
    if rec is None:
        return
    masks = rec.get("masks")
    st.session_state["undo"] = {
        "key": st.session_state.get("current_key"),
        "masks": None if masks is None else masks.copy(),
        "labels": dict(rec.get("labels", {})),
        "boxes": list(rec.get("boxes", [])),
        "boxes_display": list(rec.get("boxes_display", [])),
    }


def apply_undo(rec) -> bool:
    """Restore the undo snapshot if it's for the current image; always consume it."""
    snap = st.session_state.pop("undo", None)
    if not snap or rec is None:
        return False
    if snap.get("key") != st.session_state.get("current_key"):
        return False  # snapshot is for a different image
    rec["masks"] = snap["masks"]
    rec["labels"] = snap["labels"]
    rec["boxes"] = snap["boxes"]
    rec["boxes_display"] = snap["boxes_display"]
    return True


def reset_undo_on_navigation() -> None:
    """Drop the undo snapshot if the displayed image changed since it was taken."""
    snap = st.session_state.get("undo")
    if snap and snap.get("key") != st.session_state.get("current_key"):
        st.session_state.pop("undo", None)


def set_current_by_index(idx: int):
    ok = ordered_keys()
    if not ok:
        return
    st.session_state.current_key = ok[idx % len(ok)]


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes image intensities for Cellpose input.
    Scales mean intensity to ~127.5 or full uint8 range if mean <= 0.
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


def add_plotly_as_png_to_zip(fig_key, zip_file, out_path, default_w=900, default_h=400):
    """Adds a plotly figure stored in st.session_state[fig_key] as a PNG to the given zip file.

    Silently skips figures that were never created (e.g. when a model was uploaded
    rather than trained, so no training plots exist)."""
    fig = st.session_state.get(fig_key)
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


def plot_loss_curve(train_losses, test_losses):
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_scatter(
        x=epochs,
        y=train_losses,
        mode="lines+markers",
        name="train",
        line=dict(color="#D3E4F4", width=2),
        marker=dict(color="#D3E4F4", size=6),
    )

    # Cellpose only evaluates validation loss every 10 epochs; skip zero entries
    val_pairs = [(i + 1, v) for i, v in enumerate(test_losses) if v != 0]
    e_val = [p[0] for p in val_pairs]
    val_scores = [p[1] for p in val_pairs]
    fig.add_scatter(
        x=e_val,
        y=val_scores,
        mode="lines+markers",
        name="val",
        line=dict(color="#004280", width=2),
        marker=dict(color="#004280", size=6),
    )
    fig.update_layout(
        title="Training vs. Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        width=450,
    )
    return fig
