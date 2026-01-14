"""Segmentation and interactive mask editing for the Streamlit app."""

import hashlib
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from src.helpers.state_ops import get_current_rec
from src.helpers.classifying_functions import classes_map_from_labels, create_colour_palette
from src.helpers.state_ops import normalize_image
from src.helpers.sam2_functions import (
    segment_with_sam2,
    _clear_boxes,
    box_draw_fragment,
    integrate_new_mask,
)

ImageArray = np.ndarray  # (H, W, 3) uint8/float32
MaskArray = np.ndarray  # (H, W) int / bool
Record = dict[str, any]
Box = dict[str, float]


# -----------------------------------------------------#
# --------------- IMAGE HELPERS  --------------- #
# -----------------------------------------------------#


def create_image_mask_overlay_inner(
    image_bytes,
    mask_bytes,
    image_shape,
    mask_shape,
    classes_items,
    palette_items,
    alpha,
):
    """
    Creates image-mask overlay for display, using bytes for caching.
    """

    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)
    mask = np.frombuffer(mask_bytes, dtype=np.uint16).reshape(mask_shape)
    classes_map = dict(classes_items)
    palette = dict(palette_items)
    return create_image_mask_overlay(image, mask, classes_map, palette, alpha)


def create_image_mask_overlay(image, mask, classes_map, palette, alpha=0.5):
    """
    Create an overlay of instance masks on an image using class colours.
    image_u8:  uint8 RGB image, shape (H, W, 3)
    mask: uint{8,16,32} label image, shape (H, W), 0=background, 1..N=instances
    classes_map: dict[int -> class_name]
    palette: dict[class_name -> (r,g,b) in 0..1]
    alpha: overlay opacity for filled region
    """

    # validate inputs
    H, W = image.shape[:2]
    inst = np.asarray(mask)

    # ensure label image is same size as image
    if inst.ndim != 2:
        raise ValueError("label_inst must be a 2D label image (H, W)")
    if inst.shape != (H, W):
        # nearest to preserve integer labels
        inst = np.array(
            Image.fromarray(inst).resize((W, H), Image.NEAREST), dtype=inst.dtype
        )

    # quick exit for empty masks
    if inst.size == 0 or not np.any(inst):
        return image

    # create overlay
    out = image.astype(np.float32) / 255.0

    ids = np.unique(inst)
    ids = ids[ids != 0]  # skip background

    # draw each instance
    for iid in ids:
        # get class and colour
        cls = classes_map.get(int(iid), "__unlabeled__")
        color = np.array(palette.get(cls, palette["__unlabeled__"]), dtype=np.float32)

        # mask for this instance
        mm = inst == iid

        # filled region
        a = (mm.astype(np.float32) * alpha)[..., None]
        out = out * (1 - a) + color[None, None, :] * a

        # 1px white edge (simple interior test)
        interior = (
            mm
            & np.roll(mm, 1, 0)
            & np.roll(mm, -1, 0)
            & np.roll(mm, 1, 1)
            & np.roll(mm, -1, 1)
        )
        edge = mm & ~interior
        out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


@st.cache_data(show_spinner=False)
def cached_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    classes_map: dict,
    palette: dict,
    alpha: float,
) -> np.ndarray:
    # Convert unhashable types to something cacheable
    image_key = image.tobytes()
    mask_key = mask.tobytes()
    classes_key = tuple(sorted(classes_map.items()))
    palette_key = tuple(sorted(palette.items()))

    return create_image_mask_overlay_inner(
        image_key, mask_key, image.shape, mask.shape, classes_key, palette_key, alpha
    )


def create_image_display(rec, scale):
    disp_w, disp_h = int(rec["W"] * scale), int(rec["H"] * scale)

    mask = rec.get("masks")

    if st.session_state.get("show_overlay", False) and mask is not None and mask.any():
        labels = st.session_state.setdefault("all_classes", ["No label"])
        palette = create_colour_palette(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])

        # just shows masks over black background
        if not st.session_state.get("show_image", True):
            background = np.zeros((rec["H"], rec["W"], 3), dtype=np.uint8)

        else:
            background = rec["image"]

        base_img = cached_image_mask_overlay(
            background, rec["masks"], classes_map, palette, alpha=0.35
        )
    else:
        base_img = rec["image"]

    display_for_ui = np.array(
        Image.fromarray(base_img.astype(np.uint8)).resize(
            (disp_w, disp_h), Image.BILINEAR
        )
    )
    return base_img, display_for_ui, disp_w, disp_h


def _make_base_figure(bg_img, disp_w: int, disp_h: int, dragmode: str) -> go.Figure:
    """
    Create a Plotly figure with a background image and fixed pixel size.
    Used by both 'Draw box' and 'Draw mask' modes.
    """

    # build figure with background image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=bg_img,
            xref="x",
            yref="y",
            x=0,
            y=disp_h,
            sizex=disp_w,
            sizey=disp_h,
            sizing="stretch",
            layer="below",
        )
    )

    # set axes to image size and disable ticks
    fig.update_xaxes(visible=False, range=[0, disp_w], constrain="domain")
    fig.update_yaxes(
        visible=False,
        range=[0, disp_h],
        scaleanchor="x",
        scaleratio=1,
    )
    # set overall figure layout
    fig.update_layout(
        dragmode=dragmode,
        margin=dict(l=0, r=0, t=0, b=0),
        width=disp_w,
        height=disp_h,
    )

    return fig


def _handle_draw_mask_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Draw mask' mode."""

    # background image for plotting
    bg = Image.fromarray(display_for_ui).convert("RGBA")

    # Track display shape per record so we can clear any stale display-space data
    disp_shape = (disp_h, disp_w)
    if rec.get("display_shape") != disp_shape:
        rec["display_shape"] = disp_shape
        # reset any display-space storage tied to the old size
        st.session_state["boxes"] = []

    # unique key per image so Streamlit doesn't reuse chart state incorrectly
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_mask_{img_hash}"

    # storage for drawn boxes in display space
    if "boxes" not in st.session_state:
        st.session_state["boxes"] = []  # if you still want raw polys

    # callback to handle new lasso selections
    def update_boxes() -> None:
        event = st.session_state.get(chart_key)
        if event is None or event.selection is None:
            return

        if getattr(event.selection, "lasso", None):
            for l in event.selection.lasso:
                xs_plot = l["x"]
                ys_plot = l["y"]

                # (optional) keep polygons for debugging / later use
                clean_poly = {"x": xs_plot, "y": ys_plot}
                if clean_poly not in st.session_state["boxes"]:
                    st.session_state["boxes"].append(clean_poly)

                # Plotly coords (0 at bottom) -> display coords (0 at top)
                xs_disp = xs_plot
                ys_disp = [disp_h - y for y in ys_plot]

                # 1) polygon (display space) -> mask in display resolution
                mask_disp = polygon_xy_to_mask(xs_disp, ys_disp, disp_h, disp_w)

                # 2) resize to ORIGINAL image resolution
                mask_full = np.array(
                    Image.fromarray(mask_disp.astype(np.uint8)).resize(
                        (rec["W"], rec["H"]), Image.NEAREST
                    ),
                    dtype=bool,
                )

                # 3) integrate into rec["masks"]
                inst, new_id = integrate_new_mask(rec["masks"], mask_full)
                if new_id is not None:
                    rec["masks"] = inst
                    rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(
                        int(new_id), None
                    )

    # Build figure using the shared helper: same size as box mode
    fig = _make_base_figure(bg, disp_w, disp_h, dragmode="lasso")

    # render the plotly chart with lasso selection
    st.plotly_chart(
        fig,
        key=chart_key,
        on_select=update_boxes,
        selection_mode="lasso",
        use_container_width=False,  # same behaviour as box mode
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["lasso2d", "select2d", "zoom2d", "pan2d"],
        },
    )


def _handle_draw_box_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Draw box' mode."""

    # background image for plotting
    bg = Image.fromarray(display_for_ui).convert("RGBA")
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_{img_hash}"

    box_draw_fragment(
        bg_img=bg,
        disp_w=disp_w,
        disp_h=disp_h,
        chart_key=chart_key,
        rec=rec,
    )


def _handle_remove_mask_mode(base_img: ImageArray, disp_w: int) -> None:
    """Handle interactions when in 'Remove mask' mode."""

    streamlit_image_coordinates(
        base_img,
        key="remove_click",
        width=disp_w,
        on_click=remove_clicked,
    )


def _handle_assign_class_mode(base_img: ImageArray, disp_w: int) -> None:
    """Handle interactions when in 'Assign class' mode."""

    streamlit_image_coordinates(
        base_img,
        key="class_click",
        width=disp_w,
        on_click=assign_clicked,
    )


# -----------------------------------------------------#
# --------------- MASK HELPERS  --------------- #
# -----------------------------------------------------#


def polygon_xy_to_mask(xs, ys, height, width):
    """Rasterize a polygon given x,y coords into a (height,width) bool mask."""
    img = Image.new("L", (width, height), 0)
    xy = list(zip(xs, ys))
    ImageDraw.Draw(img).polygon(xy, outline=1, fill=1)
    return np.array(img, dtype=bool)


def remove_clicked():
    """Remove mask at clicked location."""

    # check if there was a click
    if not st.session_state["remove_click"]:
        return

    # get current record and display scale
    rec = get_current_rec()
    disp_w = st.session_state["disp_w"]

    # map click to original image coords
    s = float(disp_w / rec["W"])
    xy = (
        int(round(st.session_state["remove_click"]["x"] / s)),
        int(round(st.session_state["remove_click"]["y"] / s)),
    )

    # ignore click from previous run
    if xy == st.session_state["last_remove_xy"]:
        return

    # store last click
    st.session_state["last_remove_xy"] = xy

    # remove mask at clicked location
    x, y = xy
    m = rec.get("masks")

    iid = int(m[y, x])
    if iid == 0:
        return

    m = m.copy()
    m[m == iid] = 0
    gt = m > iid
    if gt.any():
        m[gt] -= 1

    # update record
    rec["masks"] = m
    rec["labels"] = {
        (k - 1 if k > iid else k): v
        for k, v in rec.get("labels", {}).items()
        if k != iid
    }
    st.session_state["remove_click"] = False  # prevent reprocessing on rerun


def assign_clicked():
    """Assign class to mask at clicked location."""

    # check if there was a click
    if not st.session_state["class_click"]:
        return

    # get current record and display scale
    rec = get_current_rec()
    disp_w = st.session_state["disp_w"]

    # map click to original image coords
    s = float(disp_w / rec["W"])
    xy = (
        int(round(st.session_state["class_click"]["x"] / s)),
        int(round(st.session_state["class_click"]["y"] / s)),
    )

    # ignore click from previous run
    if xy == st.session_state["last_class_xy"]:
        return

    # store last click
    st.session_state["last_class_xy"] = xy

    # assign class to mask at clicked location
    x, y = xy
    m = rec.get("masks")

    iid = int(m[y, x])
    if iid == 0:
        return

    # update label for this instance
    cur = st.session_state.get("side_current_class")
    labels = rec.setdefault("labels", {})
    if cur == "No label" or cur is None:
        labels.pop(iid, None)
    else:
        labels[iid] = cur

    st.session_state["class_click"] = False  # prevent reprocessing on rerun


# -----------------------------------------------------#
# ------------------ RENDER SIDE BAR ----------------- #
# -----------------------------------------------------#


@st.fragment
def render_cellpose_hyperparameters_fragment():
    """Render Cellpose hyperparameters editing fragment."""
    # Channel 1
    st.number_input(
        "Channel 1",
        value=st.session_state.get("cp_ch1"),
        step=1,
        format="%d",
        key="w_cp_ch1",
    )
    st.session_state["cp_ch1"] = st.session_state.get("w_cp_ch1")

    # Channel 2
    st.number_input(
        "Channel 2",
        value=st.session_state.get("cp_ch2"),
        step=1,
        format="%d",
        key="w_cp_ch2",
    )
    st.session_state["cp_ch2"] = st.session_state["w_cp_ch2"]

    # Diameter
    diam_val = st.number_input(
        "Mean cell diameter (pixels)",
        min_value=0,
        value=st.session_state.get("cp_diameter", 0),
        step=1,
        help="Leave as 0 for Cellpose to estimate diameter, or set a manual value.",
        key="w_cp_diameter",
    )
    st.session_state["cp_diameter"] = diam_val

    # cellprob threshold
    cellprob = st.number_input(
        "Cell probability threshold",
        value=float(st.session_state.get("cp_cellprob_threshold")),
        step=0.1,
        min_value=-2.0,
        max_value=2.0,
        key="w_cp_cellprob_threshold",
        help="Higher -> fewer cells.",
    )
    st.session_state["cp_cellprob_threshold"] = cellprob

    # Flow threshold
    flowthr = st.number_input(
        "Flow threshold",
        value=float(st.session_state.get("cp_flow_threshold")),
        step=0.1,
        min_value=-2.0,
        max_value=2.0,
        key="w_cp_flow_threshold",
        help="Lower -> more permissive flows.",
    )
    st.session_state["cp_flow_threshold"] = flowthr

    # Minimum size threshold
    min_size = st.number_input(
        "Minimum cell size (pixels)",
        value=int(st.session_state.get("cp_min_size")),
        min_value=0,
        step=10,
        key="w_cp_min_size",
        help="Remove masks smaller than this area.",
    )
    st.session_state["cp_min_size"] = min_size

    # Niter
    niter = st.number_input(
        "Niter",
        value=int(st.session_state["cp_niter"]),
        min_value=0,
        max_value=2000,
        step=10,
        key="w_cp_niter",
        help="Higher values favour longer, stringier, cells.",
    )
    st.session_state["cp_niter"] = niter


def render_box_tools_fragment(key_ns="side"):
    """Render SAM2 box drawing and segmentation fragment."""

    # get current record
    rec = get_current_rec()

    c1, c2 = st.columns([1, 1])
    # button to set mode to draw boxes on the image
    if c1.button(
        "Draw box",
        width='stretch',
        key=f"{key_ns}_draw_boxes",
        help="Click and drag boxes around cells",
    ):
        st.session_state["interaction_mode"] = "Draw box"
        st.rerun()
    # button to clear all boxes from the current image
    if c2.button(
        "Clear boxes",
        width='stretch',
        key="clear_boxes_button",
        help="Remove all boxes",
    ):
        _clear_boxes(rec)

    # button to segment with SAM2 the current boxes
    if st.button(
        "Generate masks from boxes",
        width='stretch',
        key=f"{key_ns}_predict",
        help="Use SAM2 to segment cells in boxes",
    ):
        # create new masks from boxes and add them to rec["mask"]
        segment_with_sam2(rec)

        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


def render_mask_tools_fragment(key_ns="side"):
    """Render manual mask drawing and removal button control fragment."""

    # get current record
    rec = get_current_rec()
    row = st.container()
    c1, c2 = row.columns([1, 1])

    # button to set mode to draw masks on the image
    if c1.button(
        "Draw mask",
        width='stretch',
        key=f"{key_ns}_draw_masks",
        help="Click and hold to draw masks",
    ):
        st.session_state["interaction_mode"] = "Draw mask"
        st.rerun()

    # button to set mode to remove masks by clicking on them
    if c2.button(
        "Remove mask",
        width='stretch',
        key=f"{key_ns}_remove_masks",
        help="Click masks to remove them",
    ):
        st.session_state["interaction_mode"] = "Remove mask"
        st.rerun()

    row = st.container()
    c1, c2 = row.columns([1, 1])

    # button to clear all masks from the current image
    if c1.button(
        "Clear masks",
        width='stretch',
        key=f"{key_ns}_clear_masks",
        help="Remove all masks from image",
    ):
        rec["masks"] = np.zeros((rec["H"], rec["W"]), dtype=np.uint16)
        rec["labels"] = {}
        rec["last_click_xy"] = None
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()

    # button to remove the last added mask
    if c2.button(
        "Undo mask",
        width='stretch',
        key=f"{key_ns}_undo_mask",
        help="Remove last mask",
    ):
        max_id = int(rec["masks"].max())
        if max_id > 0:
            rec["masks"][rec["masks"] == max_id] = 0
            rec.setdefault("labels", {}).pop(max_id, None)
        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


# -----------------------------------------------------#
# ---------------- RENDER MAIN DISPLAY --------------- #
# -----------------------------------------------------#


@st.fragment
def render_display_and_interact_fragment(key_ns="edit", scale=1.5):
    """Render main image display and interaction fragment."""

    # get current record and verify that images are uploaded
    rec = get_current_rec()

    # display image with masks overlay and interaction
    rec_for_disp = rec
    if st.session_state.get(
        "show_normalized"
    ):  # normalize background image if selected
        im = normalize_image(rec["image"])
        rec_for_disp = dict(rec)
        rec_for_disp["image"] = im

    base_img, display_for_ui, disp_w, disp_h = create_image_display(rec_for_disp, scale)
    st.session_state["disp_w"] = disp_w

    # handle interaction modes for the image (e.g. draw box, draw mask, remove mask, etc)
    mode = st.session_state.get("interaction_mode", "Draw box")  # default to draw box
    if mode == "Draw mask":
        _handle_draw_mask_mode(
            rec=rec,
            display_for_ui=display_for_ui,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Draw box":
        _handle_draw_box_mode(
            rec=rec,
            display_for_ui=display_for_ui,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Remove mask":
        _handle_remove_mask_mode(
            base_img=base_img,
            disp_w=disp_w,
        )
    elif mode == "Assign class":
        _handle_assign_class_mode(
            base_img=base_img,
            disp_w=disp_w,
        )
