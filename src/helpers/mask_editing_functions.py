"""Segmentation and interactive mask editing for the Streamlit app."""

import hashlib
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from src.helpers.state_ops import get_current_rec, snapshot_for_undo, apply_undo
from src.helpers.classifying_functions import (
    classes_map_from_labels,
    create_colour_palette,
)
from src.helpers.state_ops import normalize_image
from src.helpers.sam2_functions import (
    segment_with_sam2,
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
    Rebuild the arrays from bytes and delegate to create_image_mask_overlay.
    Assumes a uint8 image and a uint16 mask; other mask dtypes fail to reshape.
    """

    image = np.frombuffer(image_bytes, dtype=np.uint8).reshape(image_shape)
    mask = np.frombuffer(mask_bytes, dtype=np.uint16).reshape(mask_shape)
    classes_map = dict(classes_items)
    palette = dict(palette_items)
    return create_image_mask_overlay(image, mask, classes_map, palette, alpha)


def create_image_mask_overlay(image, mask, classes_map, palette, alpha=0.5):
    """
    Create an overlay of instance masks on an image using class colours.
    image: uint8 RGB image, shape (H, W, 3)
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

    out = image.astype(np.float32) / 255.0

    # per-pixel colour via a label -> colour lookup table (vectorised, no per-mask loop)
    default = np.array(palette["__unlabeled__"], dtype=np.float32)
    lut = np.tile(default, (int(inst.max()) + 1, 1))
    for iid in np.unique(inst):
        if iid:
            cls = classes_map.get(int(iid), "__unlabeled__")
            lut[iid] = palette.get(cls, default)
    color_img = lut[inst]

    # blend the filled regions in a single pass
    fg = inst != 0
    a = (fg.astype(np.float32) * alpha)[..., None]
    out = out * (1 - a) + color_img * a

    # 1px white edge: a foreground pixel whose 4-neighbour has a different label
    edge = fg & (
        (inst != np.roll(inst, 1, 0))
        | (inst != np.roll(inst, -1, 0))
        | (inst != np.roll(inst, 1, 1))
        | (inst != np.roll(inst, -1, 1))
    )
    out[edge] = 1.0

    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


# Only the current and previous mask are ever re-requested (the latter by undo).
@st.cache_data(show_spinner=False, max_entries=2)
def cached_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    classes_map: dict,
    palette: dict,
    alpha: float,
) -> np.ndarray:
    image_key = image.tobytes()
    mask_key = mask.tobytes()
    classes_key = tuple(sorted(classes_map.items()))
    palette_key = tuple(sorted(palette.items()))

    return create_image_mask_overlay_inner(
        image_key, mask_key, image.shape, mask.shape, classes_key, palette_key, alpha
    )


def create_image_display(rec, max_display_width=768):
    # Scale to fit within max_display_width while preserving aspect ratio
    scale = min(max_display_width / rec["W"], max_display_width / rec["H"])
    disp_w, disp_h = int(rec["W"] * scale), int(rec["H"] * scale)

    mask = rec.get("masks")

    if st.session_state.get("show_overlay", False) and mask is not None and mask.any():
        labels = st.session_state.setdefault("all_classes", ["No label"])
        palette = create_colour_palette(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])

        # Build the overlay at display resolution (much fewer pixels than full res).
        # This is display-only; rec["masks"] stays full-res for all measurements.
        if not st.session_state.get("show_image", True):
            background = np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
        else:
            background = np.array(
                Image.fromarray(rec["image"]).resize((disp_w, disp_h), Image.BILINEAR)
            )

        # mask is downsized (NEAREST) to the background inside the overlay helper
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
    Used by the box, freehand and ellipse drawing modes.
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


def _commit_mask(rec: Record, mask_full: MaskArray) -> None:
    """Integrate a full-resolution bool mask into rec['masks']."""
    inst, new_id = integrate_new_mask(rec["masks"], mask_full)
    if new_id is not None:
        rec["masks"] = inst
        rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(int(new_id), None)


def _lasso_select_fragment(
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
    name: str,
    on_stroke,
    show_line: bool = False,
) -> None:
    """Render a lasso-drawing Plotly canvas and dispatch each drawn stroke.

    ``on_stroke(xs, ys)`` receives one stroke's display-space coords (0 at top). Set
    ``show_line`` to strip the shaded fill so the lasso reads as a line, not a shape.
    """

    # background image for plotting
    bg = Image.fromarray(display_for_ui).convert("RGBA")

    # unique key per image so Streamlit doesn't reuse chart state incorrectly
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_{name}_{img_hash}"

    def handle() -> None:
        event = st.session_state.get(chart_key)
        sel = getattr(event, "selection", None)
        for stroke in getattr(sel, "lasso", None) or []:
            # Plotly coords (0 at bottom) -> display coords (0 at top)
            on_stroke(stroke["x"], [disp_h - y for y in stroke["y"]])

    fig = _make_base_figure(bg, disp_w, disp_h, dragmode="lasso")
    if show_line:
        fig.update_layout(
            newselection=dict(line=dict(color="red", width=1)),
            activeselection=dict(fillcolor="rgba(0,0,0,0)"),
        )

    st.plotly_chart(
        fig,
        key=chart_key,
        on_select=handle,
        selection_mode="lasso",
        width="content",
        config={
            # Plotly zoom/pan disabled so the chart doesn't reset zoom on every
            # image refresh; use the browser's own pinch-zoom instead.
            "scrollZoom": False,
            "displaylogo": False,
            "modeBarButtons": [["lasso2d"]],
        },
    )


def _handle_draw_mask_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Freehand' mask drawing mode."""

    def draw(xs, ys) -> None:
        snapshot_for_undo(rec)
        mask_full = polygon_xy_to_full_mask(xs, ys, disp_w, disp_h, rec["W"], rec["H"])
        _commit_mask(rec, mask_full)

    _lasso_select_fragment(display_for_ui, disp_w, disp_h, key_ns, "mask", draw)


def _handle_draw_ellipse_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Ellipse' mask drawing mode.

    The user drags a box around a colony; the bounding box is filled with an
    ellipse (a circle when the drag is square). The shape is committed the
    moment the drag ends, so its position and size match exactly what was
    dragged with no follow-up adjustment.
    """

    # background image for plotting
    bg = Image.fromarray(display_for_ui).convert("RGBA")

    # unique key per image so Streamlit doesn't reuse chart state incorrectly
    img_hash = hashlib.md5(bg.tobytes()).hexdigest()[:8]
    chart_key = f"{key_ns}_plotly_ellipse_{img_hash}"

    # callback to turn each box-drag into a filled ellipse
    def add_ellipse() -> None:
        event = st.session_state.get(chart_key)
        sel = getattr(event, "selection", None)
        for b in getattr(sel, "box", None) or []:
            x0, x1 = sorted(map(float, b["x"]))
            y0, y1 = sorted(map(float, b["y"]))
            if x1 <= x0 or y1 <= y0:  # ignore zero-size drags
                continue
            snapshot_for_undo(rec)
            # display box -> full-res box (Plotly y 0 at bottom -> display y 0 at top)
            sx, sy = rec["W"] / disp_w, rec["H"] / disp_h
            mask_full = ellipse_box_to_mask(
                x0 * sx, (disp_h - y1) * sy, x1 * sx, (disp_h - y0) * sy, rec["H"], rec["W"]
            )
            _commit_mask(rec, mask_full)

    # Build figure using the shared helper: same size as box mode
    fig = _make_base_figure(bg, disp_w, disp_h, dragmode="select")

    # render the plotly chart with box selection
    st.plotly_chart(
        fig,
        key=chart_key,
        on_select=add_ellipse,
        selection_mode="box",
        width="content",
        config={
            # Plotly zoom/pan disabled so the chart doesn't reset zoom on every
            # image refresh; use the browser's own pinch-zoom instead.
            "scrollZoom": False,
            "displaylogo": False,
            "modeBarButtons": [["select2d"]],
        },
    )


def _handle_cut_mask_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Split masks' mode.

    The user clicks and drags to draw a freehand line; any mask the line passes
    all the way through is split into separate masks along that line.
    """

    def cut(xs, ys) -> None:
        barrier = polyline_xy_to_barrier(xs, ys, disp_w, disp_h, rec["W"], rec["H"])
        cut_masks_along_barrier(rec, barrier)

    _lasso_select_fragment(
        display_for_ui, disp_w, disp_h, key_ns, "cut", cut, show_line=True
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


def _draw_boxes_on(img: ImageArray, rec: Record) -> ImageArray:
    """Return a copy of img with the record's boxes drawn as red rectangles.

    Boxes are stored in full-res image coords; they're scaled to img's size so
    they can be seen and clicked in 'Remove' mode."""
    boxes = rec.get("boxes", [])
    if not boxes:
        return img
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    h, w = img.shape[:2]
    sx, sy = w / rec["W"], h / rec["H"]
    for x0, y0, x1, y1 in boxes:
        d.rectangle([x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=(255, 0, 0), width=2)
    return np.array(im)


def _refocus_main_document() -> None:
    """Refocus the parent document so button shortcuts work after an iframe click.

    The nonce forces the shim to re-run each rerun (identical HTML wouldn't)."""
    nonce = st.session_state.get("_refocus_nonce", 0) + 1
    st.session_state["_refocus_nonce"] = nonce
    components.html(
        f"<script>window.parent.focus();</script><!--{nonce}-->",
        height=0,
    )


def _handle_remove_mask_mode(base_img: ImageArray, disp_w: int, rec: Record) -> None:
    """Handle interactions when in 'Remove' mode (click a mask or box to delete it)."""

    streamlit_image_coordinates(
        _draw_boxes_on(base_img, rec),
        key="remove_click",
        width=disp_w,
        on_click=remove_clicked,
    )
    _refocus_main_document()


def _handle_assign_class_mode(base_img: ImageArray, disp_w: int) -> None:
    """Handle interactions when in 'Assign class' mode."""

    streamlit_image_coordinates(
        base_img,
        key="class_click",
        width=disp_w,
        on_click=assign_clicked,
    )
    _refocus_main_document()


def _handle_merge_mask_mode(
    rec: Record,
    display_for_ui: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Merge masks' mode.

    The user draws a lasso; every mask lying completely inside the selection is
    merged with any other selected mask it touches (touching masks form groups).
    """

    def merge(xs, ys) -> None:
        region = polygon_xy_to_full_mask(xs, ys, disp_w, disp_h, rec["W"], rec["H"])
        merge_in_lasso(rec, region)

    _lasso_select_fragment(display_for_ui, disp_w, disp_h, key_ns, "merge", merge)


# -----------------------------------------------------#
# --------------- MASK HELPERS  --------------- #
# -----------------------------------------------------#


def polygon_xy_to_mask(xs, ys, height, width):
    """Rasterize a polygon given x,y coords into a (height,width) bool mask."""
    img = Image.new("L", (width, height), 0)
    xy = list(zip(xs, ys))
    ImageDraw.Draw(img).polygon(xy, outline=1, fill=1)
    return np.array(img, dtype=bool)


def ellipse_box_to_mask(x0, y0, x1, y1, height, width):
    """Rasterize a filled ellipse inscribed in the (x0,y0,x1,y1) bounding box."""
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).ellipse([x0, y0, x1, y1], outline=1, fill=1)
    return np.array(img, dtype=bool)


def polyline_xy_to_barrier(xs, ys, disp_w, disp_h, width, height):
    """Rasterize a display-space freehand line into a thin (height,width) bool barrier."""
    sx, sy = width / disp_w, height / disp_h
    pts = [(x * sx, y * sy) for x, y in zip(xs, ys)]
    img = Image.new("L", (width, height), 0)
    if len(pts) >= 2:
        # scale line thickness with resolution so the cut always separates regions
        thickness = max(3, round(3 * width / disp_w))
        ImageDraw.Draw(img).line(pts, fill=1, width=thickness)
    return np.array(img, dtype=bool)


def polygon_xy_to_full_mask(xs, ys, disp_w, disp_h, width, height):
    """Rasterize a display-space polygon into a filled full-res (height,width) bool mask."""
    sx, sy = width / disp_w, height / disp_h
    return polygon_xy_to_mask([x * sx for x in xs], [y * sy for y in ys], height, width)


def cut_masks_along_barrier(rec: Record, barrier: MaskArray) -> None:
    """Split any mask that the barrier line fully dissects into separate masks.

    For each instance the barrier touches, the barrier pixels are removed and the
    remainder relabelled: only if this yields >=2 connected components did the line
    pass all the way through, so those instances are split (each new piece inherits
    the original class). Barrier pixels are then reassigned to their nearest piece so
    the resulting masks touch with no gap. Partially-crossed masks stay a single
    component and are left untouched.
    """
    m = rec["masks"]
    touched = np.unique(m[barrier])
    touched = touched[touched != 0]
    snapshot_for_undo(rec)

    struct = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labels = rec.setdefault("labels", {})
    out = m
    next_id = int(m.max())
    changed = False
    split_n = 0  # how many masks the line fully dissected

    for iid in map(int, touched):
        # work within the instance's bounding box for efficiency
        ys, xs = np.where(m == iid)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        full = m[y0:y1, x0:x1] == iid
        pieces, n = ndi.label(full & ~barrier[y0:y1, x0:x1], structure=struct)
        if n < 2:
            continue  # line does not fully dissect this mask -> leave intact

        if not changed:
            out = m.copy()
            changed = True
        split_n += 1
        if next_id + (n - 1) > np.iinfo(out.dtype).max:
            out = out.astype(np.uint32)

        # assign every pixel (incl. barrier) to its nearest piece so masks abut
        nearest = ndi.distance_transform_edt(
            pieces == 0, return_indices=True, return_distances=False
        )
        filled = pieces[tuple(nearest)]
        sub = out[y0:y1, x0:x1]
        # piece 1 already keeps iid (out is a copy of m); only relabel the rest
        for k in range(2, n + 1):
            pid = next_id + (k - 1)
            sub[full & (filled == k)] = pid
            labels[pid] = labels.get(iid)
        next_id += n - 1

    if changed:
        rec["masks"] = out
    st.toast(f"Split {split_n} mask{'s' if split_n != 1 else ''}.")


def remove_clicked():
    """Remove the mask or box at the clicked location."""

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

    # remove the box or mask at the clicked location (boxes take priority)
    x, y = xy

    # remove the first box that contains the click, if any
    for i, (bx0, by0, bx1, by1) in enumerate(rec.get("boxes", [])):
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            snapshot_for_undo(rec)
            rec["boxes"].pop(i)
            if i < len(rec.get("boxes_display", [])):
                rec["boxes_display"].pop(i)
            st.session_state["remove_click"] = False
            return

    # no box here -> remove the mask at the click and compact the ids above it
    m = rec.get("masks")
    iid = int(m[y, x]) if m is not None else 0
    if iid != 0:
        snapshot_for_undo(rec)
        m = m.copy()
        m[m == iid] = 0
        gt = m > iid
        if gt.any():
            m[gt] -= 1
        rec["masks"] = m
        rec["labels"] = {
            (k - 1 if k > iid else k): v
            for k, v in rec.get("labels", {}).items()
            if k != iid
        }

    st.session_state["remove_click"] = False  # prevent reprocessing on rerun


def merge_in_lasso(rec: Record, region: MaskArray) -> None:
    """Merge groups of touching masks that lie completely inside the lasso region.

    Only masks with no pixel outside the selection are considered; among those, any
    that touch (8-connectivity) are merged together, each merged mask keeping the
    lowest id. Instance ids and labels are then compacted.
    """
    m = rec["masks"]
    inside = set(np.unique(m[region])) - {0}
    outside = set(np.unique(m[~region]))
    selected = inside - outside  # masks lying completely within the selection
    snapshot_for_undo(rec)

    merged_n = 0  # how many masks were combined
    groups = 0  # how many merged masks they became
    if len(selected) >= 2:
        # group selected masks by contact: touching masks share a connected component
        picked = np.isin(m, np.fromiter(selected, dtype=m.dtype))
        comp, n = ndi.label(picked, structure=np.ones((3, 3), dtype=bool))

        out = m.copy()
        for c in range(1, n + 1):
            cm = comp == c
            ids = np.unique(m[cm])
            if ids.size > 1:  # >=2 selected masks touch here -> merge them
                out[cm] = ids.min()
                merged_n += int(ids.size)
                groups += 1

        if merged_n:
            # compact ids to a contiguous range and remap labels to match
            uniq, inv = np.unique(out, return_inverse=True)
            rec["masks"] = inv.reshape(m.shape).astype(m.dtype)
            remap = {int(old): new for new, old in enumerate(uniq)}
            rec["labels"] = {
                remap[k]: v for k, v in rec.get("labels", {}).items() if remap.get(k, 0)
            }

    st.toast(f"Merged {merged_n} masks in {groups}.")


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

    snapshot_for_undo(rec)
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
    # Diameter
    diam_val = st.number_input(
        "Mean cell diameter (pixels)",
        min_value=0,
        value=int(st.session_state.get("cp_diameter", 0)),
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

    # button to set mode to draw boxes on the image
    if st.button(
        "Draw box",
        width="stretch",
        key=f"{key_ns}_draw_boxes",
        shortcut="B",
        help="Click and drag boxes around cells (shortcut: B)",
    ):
        st.session_state["interaction_mode"] = "Draw box"
        st.rerun()

    # button to segment with SAM2 the current boxes
    if st.button(
        "Generate masks from boxes",
        width="stretch",
        key=f"{key_ns}_predict",
        shortcut="G",
        help="Use SAM2 to segment cells in boxes (shortcut: G)",
    ):
        # create new masks from boxes and add them to rec["mask"]
        segment_with_sam2(rec)

        st.session_state["pred_canvas_nonce"] += 1
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()


def render_draw_mask_tools_fragment(key_ns="side"):
    """Render the manual mask drawing mode options (Freehand / Ellipse)."""

    c1, c2 = st.columns([1, 1])

    # button to set mode to freehand (lasso) mask drawing
    if c1.button(
        "Freehand",
        width="stretch",
        key=f"{key_ns}_draw_masks",
        shortcut="F",
        help="Click and hold to draw a freehand mask (shortcut: F)",
    ):
        st.session_state["interaction_mode"] = "Freehand"
        st.rerun()

    # button to set mode to ellipse mask drawing
    if c2.button(
        "Ellipse",
        width="stretch",
        key=f"{key_ns}_draw_ellipse",
        shortcut="E",
        help="Drag a box around a colony to fill it with a rough ellipse (shortcut: E)",
    ):
        st.session_state["interaction_mode"] = "Ellipse"
        st.rerun()

    c3, c4 = st.columns([1, 1])

    # button to set mode to cutting masks with a drawn line
    if c3.button(
        "Split masks",
        width="stretch",
        key=f"{key_ns}_cut_mask",
        shortcut="S",
        help="Click and drag a line all the way through a mask to split it in two (shortcut: S)",
    ):
        st.session_state["interaction_mode"] = "Split masks"
        st.rerun()

    # button to set mode to merging two touching masks
    if c4.button(
        "Merge masks",
        width="stretch",
        key=f"{key_ns}_merge_masks",
        shortcut="M",
        help="Draw a lasso around touching masks to merge them into one (shortcut: M)",
    ):
        st.session_state["interaction_mode"] = "Merge masks"
        st.rerun()


def render_common_tools_fragment(key_ns="tools"):
    """Render the always-visible editing tools (Remove, Clear All, Undo), shown
    outside the segmentation/classification tabs so they're available in both."""

    # get current record
    rec = get_current_rec()

    # button to set mode to remove masks or boxes by clicking on them
    if st.button(
        "Remove",
        width="stretch",
        key=f"{key_ns}_remove",
        shortcut="D",
        help="Click masks or boxes to remove them (shortcut: D)",
    ):
        st.session_state["interaction_mode"] = "Remove"
        st.rerun()

    row = st.container()
    c1, c2 = row.columns([1, 1])

    # button to clear all masks and boxes from the current image
    if c1.button(
        "Clear All",
        width="stretch",
        key=f"{key_ns}_clear_all",
        help="Remove all masks and boxes from this image",
    ):
        snapshot_for_undo(rec)
        rec["masks"] = np.zeros((rec["H"], rec["W"]), dtype=np.uint16)
        rec["labels"] = {}
        rec["boxes"] = []
        rec["boxes_display"] = []
        rec["last_click_xy"] = None
        st.session_state["edit_canvas_nonce"] += 1
        st.rerun()

    # single-level undo of the last action
    render_undo_button(c2, key_ns=key_ns)


def render_undo_button(container=st, key_ns="side"):
    """Render the undo button, reverting the last mask/box action via apply_undo
    (single-level; only the most recent action is recoverable)."""
    rec = get_current_rec()
    if container.button(
        "Undo",
        width="stretch",
        key=f"{key_ns}_undo",
        shortcut="ctrl+z",
        help="Undo the last action — only the most recent action can be undone (shortcut: Ctrl+Z)",
    ):
        if apply_undo(rec):
            st.session_state["pred_canvas_nonce"] += 1
            st.session_state["edit_canvas_nonce"] += 1
            st.rerun()


# -----------------------------------------------------#
# ---------------- RENDER MAIN DISPLAY --------------- #
# -----------------------------------------------------#


@st.fragment
def render_display_and_interact_fragment(key_ns="edit", max_display_width=768):
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

    base_img, display_for_ui, disp_w, disp_h = create_image_display(
        rec_for_disp, max_display_width
    )
    st.session_state["disp_w"] = disp_w

    # handle interaction modes for the image (e.g. draw box, draw mask, remove mask, etc)
    mode = st.session_state.get("interaction_mode", "Draw box")  # default to draw box
    if mode == "Freehand":
        _handle_draw_mask_mode(
            rec=rec,
            display_for_ui=display_for_ui,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Ellipse":
        _handle_draw_ellipse_mode(
            rec=rec,
            display_for_ui=display_for_ui,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Split masks":
        _handle_cut_mask_mode(
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
    elif mode == "Merge masks":
        _handle_merge_mask_mode(
            rec=rec,
            display_for_ui=display_for_ui,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Remove":
        _handle_remove_mask_mode(
            base_img=base_img,
            disp_w=disp_w,
            rec=rec,
        )
    elif mode == "Assign class":
        _handle_assign_class_mode(
            base_img=base_img,
            disp_w=disp_w,
        )
