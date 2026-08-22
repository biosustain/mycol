from contextlib import nullcontext
import numpy as np
import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from scipy import ndimage as ndi

from src.helpers.state_ops import snapshot_for_undo, disp_to_full, full_to_disp
from src.helpers.plot_helpers import MIN_DRAG_PX, make_base_figure

ss = st.session_state


# new masks lose priority where they overlap existing ones, so a mask can be cut
# into pieces on insert; integrate_new_mask uses this to keep only the largest.
def keep_largest_part(mask: np.ndarray) -> np.ndarray:
    """Return only the largest connected component of a boolean mask."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    lab, n = ndi.label(mask)
    if n == 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def integrate_new_mask(original: np.ndarray, new_binary: np.ndarray):
    """
    Add a new mask into a label image.
    - original: (H,W) int labels, 0=background, 1..N instances
    - new_binary: (H,W) boolean mask
    Returns (updated_label_image, new_id or None)
    """
    out = original
    nb = new_binary.astype(bool)
    if nb.ndim != 2 or not nb.any():
        return out, None

    # write only where background
    write = (out == 0) & nb
    if not write.any():
        return out, None

    max_id = int(out.max(initial=0))
    new_id = max_id + 1

    # upcast if needed
    if new_id > np.iinfo(out.dtype).max:
        out = out.astype(np.uint32)
    else:
        out = out.copy()

    out[write] = new_id

    # --- check contiguity: keep only the largest surviving component ---
    mask_new = out == new_id
    mask_new = keep_largest_part(mask_new)
    if not mask_new.any():
        return original, None  # nothing left after check

    out[out == new_id] = 0  # clear possibly cut version
    out[mask_new] = new_id  # reapply only largest part

    return out, new_id


def _update_boxes(chart_key: str, rec: dict):
    """Callback run when a selection is made on the Plotly chart."""
    event = ss.get(chart_key)
    sel = getattr(event, "selection", None)
    if not sel or not sel.box:
        return

    boxes = rec.setdefault("boxes", [])

    disp_h = ss.get("disp_h")
    if not disp_h or not rec.get("W"):
        return

    for b in sel.box:
        x0_plot, x1_plot = map(float, b["x"])
        y0_plot, y1_plot = map(float, b["y"])

        # stray click, not a box
        if abs(x1_plot - x0_plot) < MIN_DRAG_PX or abs(y1_plot - y0_plot) < MIN_DRAG_PX:
            continue

        # Normalize ordering
        if x1_plot < x0_plot:
            x0_plot, x1_plot = x1_plot, x0_plot
        if y1_plot < y0_plot:
            y0_plot, y1_plot = y1_plot, y0_plot

        # Flip Y (Plotly 0 at bottom -> display 0 at top), then display -> full-res
        fx0, fy0 = disp_to_full(x0_plot, disp_h - y1_plot)
        fx1, fy1 = disp_to_full(x1_plot, disp_h - y0_plot)

        # Clamp to image bounds
        x0 = max(0, min(rec["W"] - 1, int(round(fx0))))
        x1 = max(0, min(rec["W"], int(round(fx1))))
        y0 = max(0, min(rec["H"] - 1, int(round(fy0))))
        y1 = max(0, min(rec["H"], int(round(fy1))))

        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        box_orig = (x0, y0, x1, y1)
        if box_orig not in boxes:
            snapshot_for_undo(rec)
            boxes.append(box_orig)


def _make_figure_with_boxes(bg_img, disp_w, disp_h, rec: dict):
    """Create a Plotly figure with background image and drawn boxes overlayed."""

    # create base figure
    fig = make_base_figure(bg_img, disp_w, disp_h, dragmode="select")

    # full-res boxes -> display coords, then Plotly display coords (y bottom)
    display_boxes = []
    for x0, y0, x1, y1 in rec.get("boxes", []):
        dx0, dy0 = full_to_disp(x0, y0)
        dx1, dy1 = full_to_disp(x1, y1)
        display_boxes.append(
            {"x0": dx0, "x1": dx1, "y0": disp_h - dy0, "y1": disp_h - dy1}
        )

    # add boxes
    for box in display_boxes:
        fig.add_shape(
            type="rect",
            x0=box["x0"],
            x1=box["x1"],
            y0=box["y0"],
            y1=box["y1"],
            line=dict(color="red", width=2),
            fillcolor="rgba(255,0,0,0.15)",
            layer="above",
        )

    return fig


def _clear_boxes(rec: dict):
    """Clear all boxes for the current record."""
    rec["boxes"] = []


@st.fragment
def box_draw_fragment(bg_img, disp_w, disp_h, chart_key: str, rec: dict):
    """Render the Plotly chart for 'Draw box' mode, with box selection handling."""
    fig = _make_figure_with_boxes(bg_img, disp_w, disp_h, rec)
    st.plotly_chart(
        fig,
        key=chart_key,
        selection_mode="box",
        on_select=lambda: _update_boxes(chart_key, rec),
        width="content",  # respects fig.width / fig.height
        config={
            # Plotly zoom/pan disabled so the chart doesn't reset zoom on every
            # image refresh; use the browser's own pinch-zoom instead.
            "scrollZoom": False,
            "displaylogo": False,
            "modeBarButtons": [["select2d"]],
        },
    )


def prep_image_for_sam(img: np.ndarray) -> np.ndarray:
    """Return img as (H, W, 3) uint8, the form the predictor expects"""
    a = img
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = np.array(Image.fromarray(a).convert("RGB"))

    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        mx = float(a.max()) if a.size else 1.0
        if mx <= 1.0:
            a *= 255.0
        elif mx > 255.0:
            a *= 255.0 / mx
        a = np.clip(a, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(a)


@st.cache_resource(show_spinner="Loading MobileSAM weights…")
def _load_box_segmenter():
    """Load MobileSAM once and reuse it across reruns"""
    from mobile_sam import SamPredictor, sam_model_registry

    device = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    ckpt = hf_hub_download("dhkim2810/MobileSAM", "mobile_sam.pt")
    sam = sam_model_registry["vit_t"](checkpoint=ckpt).to(device).eval()

    return SamPredictor(sam), device


def segment_boxes(rec: dict):
    """Segment the cells in rec['boxes'] with MobileSAM, merging each into rec['masks'].

    The record's boxes are cleared once their masks have been integrated."""

    boxes = np.asarray(rec.get("boxes", []), dtype=np.float32)
    if boxes.size == 0:
        st.info("No boxes drawn yet.")
        return

    # covers the box clear below as well as the mask edits
    snapshot_for_undo(rec)

    predictor, device = _load_box_segmenter()
    img = prep_image_for_sam(rec["image"])
    amp = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), amp:
        predictor.set_image(img)

        # batched so a large set of boxes cannot exhaust memory
        for start in range(0, len(boxes), 8):
            batch = torch.as_tensor(boxes[start : start + 8], device=device)
            batch = predictor.transform.apply_boxes_torch(batch, img.shape[:2])
            masks, scores, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=batch,
                multimask_output=True,
            )

            # pick the best of each box's three masks before leaving the device
            rows = torch.arange(len(masks), device=masks.device)
            best = masks[rows, scores.argmax(-1)].cpu().numpy()

            for mask in best:
                inst, new_id = integrate_new_mask(rec["masks"], mask)
                if new_id is not None:
                    rec["masks"] = inst
                    rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                        int(new_id), None
                    )

    _clear_boxes(rec)
