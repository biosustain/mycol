"""Segmentation and interactive mask editing for the Streamlit app."""

import hashlib
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

from src.helpers.state_ops import (
    get_current_rec,
    snapshot_for_undo,
    apply_undo,
    disp_to_full,
    full_to_disp,
)
from src.helpers.classifying_functions import (
    classes_map_from_labels,
    create_colour_palette,
)
from src.helpers.state_ops import normalize_image
from src.helpers.plot_helpers import MIN_DRAG_PX, make_base_figure
from src.helpers.sam2_functions import (
    segment_with_sam2,
    box_draw_fragment,
    integrate_new_mask,
)

ss = st.session_state

ImageArray = np.ndarray  # (H, W, 3) uint8/float32
MaskArray = np.ndarray  # (H, W) int / bool
Record = dict[str, any]
Box = dict[str, float]


# -----------------------------------------------------#
# --------------- IMAGE HELPERS  --------------- #
# -----------------------------------------------------#


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


# Caches the current and previous mask overlay.
@st.cache_data(show_spinner=False, max_entries=2)
def cached_image_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    classes_map: dict,
    palette: dict,
    alpha: float,
) -> np.ndarray:
    return create_image_mask_overlay(image, mask, classes_map, palette, alpha)


def create_image_display(rec, viewport=800):
    """Render a zoomed crop of the image (with overlay) into a fixed-size viewport.

    The visible region is a `zoom`-times window centred on the pan point; the
    on-screen size stays constant. ss["view"] = (ox, oy, s) records the crop so
    interactions can map display coords back to full resolution. At zoom 1 the crop
    is the whole image, i.e. identical to the original whole-image display."""
    W, H = rec["W"], rec["H"]
    zoom = max(1.0, float(ss.get("zoom", 1.0)))

    # visible crop in full-res px, centred on the pan point and clamped to the image
    cw, ch = W / zoom, H / zoom
    cx = min(max(float(ss.get("pan_cx", W / 2)), cw / 2), W - cw / 2)
    cy = min(max(float(ss.get("pan_cy", H / 2)), ch / 2), H - ch / 2)
    cw, ch = int(round(cw)), int(round(ch))
    ox = max(0, min(int(round(cx - cw / 2)), W - cw))
    oy = max(0, min(int(round(cy - ch / 2)), H - ch))

    # constant on-screen size (fits the whole image at zoom 1)
    fit = min(viewport / W, viewport / H)
    disp_w, disp_h = max(1, int(W * fit)), max(1, int(H * fit))
    ss["view"] = (ox, oy, cw / disp_w)  # (crop origin, full-res px per display px)
    ss["disp_h"] = disp_h  # read by the box-drawing callback to flip Plotly's y axis

    img_crop = rec["image"][oy : oy + ch, ox : ox + cw]
    bg_disp = np.array(Image.fromarray(img_crop).resize((disp_w, disp_h), Image.BILINEAR))

    mask = rec.get("masks")
    if ss.get("show_overlay", False) and mask is not None and mask.any():
        labels = ss.setdefault("all_classes", ["No label"])
        palette = create_colour_palette(labels)
        classes_map = classes_map_from_labels(rec["masks"], rec["labels"])
        background = (
            np.zeros((disp_h, disp_w, 3), dtype=np.uint8)
            if not ss.get("show_image", True)
            else bg_disp
        )
        # the cropped mask is downsized (NEAREST) to the background in the overlay helper
        base_img = cached_image_mask_overlay(
            background, mask[oy : oy + ch, ox : ox + cw], classes_map, palette, alpha=0.35
        )
    else:
        base_img = bg_disp

    return base_img, disp_w, disp_h


def _commit_mask(rec: Record, mask_full: MaskArray) -> None:
    """Integrate a full-resolution bool mask into rec['masks']."""
    inst, new_id = integrate_new_mask(rec["masks"], mask_full)
    if new_id is not None:
        rec["masks"] = inst
        rec.setdefault("labels", {})[int(new_id)] = rec["labels"].get(int(new_id), None)


def _chart_bg(base_img: ImageArray, key_ns: str, name: str) -> tuple[Image.Image, str]:
    """Plotly background image plus its chart key.

    The key carries an image hash so Streamlit doesn't reuse chart state across images."""
    bg = Image.fromarray(base_img).convert("RGBA")
    return bg, f"{key_ns}_plotly_{name}_{hashlib.md5(bg.tobytes()).hexdigest()[:8]}"


def _selection_of(chart_key: str, kind: str) -> list:
    """The chart's ``"lasso"`` or ``"box"`` selections, or an empty list."""
    sel = getattr(ss.get(chart_key), "selection", None)
    return getattr(sel, kind, None) or []


def _selection_chart(
    bg: Image.Image,
    disp_w: int,
    disp_h: int,
    chart_key: str,
    mode: str,
    handler,
    show_line: bool = False,
) -> None:
    """Render a Plotly canvas that reports selections back through ``handler``.

    ``mode`` is "lasso" or "box"; ``show_line`` strips the shaded fill so a lasso
    reads as a line, not a shape."""
    is_lasso = mode == "lasso"
    fig = make_base_figure(bg, disp_w, disp_h, dragmode="lasso" if is_lasso else "select")
    if show_line:
        fig.update_layout(
            newselection=dict(line=dict(color="red", width=1)),
            activeselection=dict(fillcolor="rgba(0,0,0,0)"),
        )

    st.plotly_chart(
        fig,
        key=chart_key,
        on_select=handler,
        selection_mode=mode,
        width="content",
        config={
            # Plotly zoom/pan disabled so the chart doesn't reset zoom on every
            # image refresh; use the browser's own pinch-zoom instead.
            "scrollZoom": False,
            "displaylogo": False,
            "modeBarButtons": [["lasso2d" if is_lasso else "select2d"]],
        },
    )


def _lasso_chart(
    base_img: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
    name: str,
    on_stroke,
    show_line: bool = False,
) -> None:
    """Lasso canvas that dispatches each drawn stroke to ``on_stroke(xs, ys)``.

    Strokes arrive in display-space coords, 0 at the top."""
    bg, chart_key = _chart_bg(base_img, key_ns, name)

    def handle() -> None:
        for stroke in _selection_of(chart_key, "lasso"):
            # Plotly coords (0 at bottom) -> display coords (0 at top)
            on_stroke(stroke["x"], [disp_h - y for y in stroke["y"]])

    _selection_chart(bg, disp_w, disp_h, chart_key, "lasso", handle, show_line)


def _handle_draw_mask_mode(
    rec: Record,
    base_img: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Freehand' mask drawing mode."""

    def draw(xs, ys) -> None:
        snapshot_for_undo(rec)
        mask_full = polygon_xy_to_full_mask(xs, ys, rec["H"], rec["W"])
        _commit_mask(rec, mask_full)

    _lasso_chart(base_img, disp_w, disp_h, key_ns, "mask", draw)


def _handle_draw_ellipse_mode(
    rec: Record,
    base_img: ImageArray,
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
    bg, chart_key = _chart_bg(base_img, key_ns, "ellipse")

    # callback to turn each box-drag into a filled ellipse
    def add_ellipse() -> None:
        for b in _selection_of(chart_key, "box"):
            x0, x1 = sorted(map(float, b["x"]))
            y0, y1 = sorted(map(float, b["y"]))
            if x1 - x0 < MIN_DRAG_PX or y1 - y0 < MIN_DRAG_PX:  # stray click
                continue
            snapshot_for_undo(rec)
            # display box -> full-res box (Plotly y 0 at bottom -> display y 0 at top)
            fx0, fy0 = disp_to_full(x0, disp_h - y1)
            fx1, fy1 = disp_to_full(x1, disp_h - y0)
            mask_full = ellipse_box_to_mask(fx0, fy0, fx1, fy1, rec["H"], rec["W"])
            _commit_mask(rec, mask_full)

    _selection_chart(bg, disp_w, disp_h, chart_key, "box", add_ellipse)


def _handle_cut_mask_mode(
    rec: Record,
    base_img: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Split masks' mode.

    The user clicks and drags to draw a freehand line; any mask the line passes
    all the way through is split into separate masks along that line.
    """

    def cut(xs, ys) -> None:
        barrier = polyline_xy_to_barrier(xs, ys, rec["H"], rec["W"])
        cut_masks_along_barrier(rec, barrier)

    _lasso_chart(base_img, disp_w, disp_h, key_ns, "cut", cut, show_line=True)


def _handle_draw_box_mode(
    rec: Record,
    base_img: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Draw box' mode."""
    bg, chart_key = _chart_bg(base_img, key_ns, "box")

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
    for x0, y0, x1, y1 in boxes:
        dx0, dy0 = full_to_disp(x0, y0)
        dx1, dy1 = full_to_disp(x1, y1)
        d.rectangle([dx0, dy0, dx1, dy1], outline=(255, 0, 0), width=2)
    return np.array(im)


def _refocus_main_document() -> None:
    """Refocus the parent document so button shortcuts work after an iframe click.

    The nonce forces the shim to re-run each rerun (identical HTML wouldn't)."""
    nonce = ss.get("_refocus_nonce", 0) + 1
    ss["_refocus_nonce"] = nonce
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


def _handle_join_mask_mode(
    rec: Record,
    base_img: ImageArray,
    disp_w: int,
    disp_h: int,
    key_ns: str,
) -> None:
    """Handle interactions when in 'Join masks' mode.

    The user draws a lasso; every mask lying completely inside the selection is
    joined with any other selected mask it touches (touching masks form groups).
    """

    def join(xs, ys) -> None:
        region = polygon_xy_to_full_mask(xs, ys, rec["H"], rec["W"])
        join_in_lasso(rec, region)

    _lasso_chart(base_img, disp_w, disp_h, key_ns, "join", join)


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


def polyline_xy_to_barrier(xs, ys, height, width):
    """Rasterize a display-space freehand line into a thin (height,width) bool barrier."""
    pts = [disp_to_full(x, y) for x, y in zip(xs, ys)]
    img = Image.new("L", (width, height), 0)
    if len(pts) >= 2:
        # thickness scales with the display->full ratio so the cut always separates regions
        thickness = max(3, round(3 * ss["view"][2]))
        ImageDraw.Draw(img).line(pts, fill=1, width=thickness)
    return np.array(img, dtype=bool)


def polygon_xy_to_full_mask(xs, ys, height, width):
    """Rasterize a display-space polygon into a filled full-res (height,width) bool mask."""
    pts = [disp_to_full(x, y) for x, y in zip(xs, ys)]
    return polygon_xy_to_mask([p[0] for p in pts], [p[1] for p in pts], height, width)


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
            # first mask is about to be split -> take the single undo snapshot now
            snapshot_for_undo(rec)
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
    if not ss["remove_click"]:
        return

    # get current record
    rec = get_current_rec()

    # map click (viewport) to full-res image coords
    fx, fy = disp_to_full(ss["remove_click"]["x"], ss["remove_click"]["y"])
    xy = (
        int(min(max(round(fx), 0), rec["W"] - 1)),
        int(min(max(round(fy), 0), rec["H"] - 1)),
    )

    # ignore click from previous run
    if xy == ss["last_remove_xy"]:
        return

    # store last click
    ss["last_remove_xy"] = xy

    # remove the box or mask at the clicked location (boxes take priority)
    x, y = xy

    # remove the first box that contains the click, if any
    for i, (bx0, by0, bx1, by1) in enumerate(rec.get("boxes", [])):
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            snapshot_for_undo(rec)
            rec["boxes"].pop(i)
            ss["remove_click"] = False
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

    ss["remove_click"] = False  # prevent reprocessing on rerun


def join_in_lasso(rec: Record, region: MaskArray) -> None:
    """Join groups of touching masks that lie completely inside the lasso region.

    Only masks with no pixel outside the selection are considered; among those, any
    that touch (8-connectivity) are joined together, each joined mask keeping the
    lowest id. Instance ids and labels are then compacted.
    """
    m = rec["masks"]
    inside = set(np.unique(m[region])) - {0}
    outside = set(np.unique(m[~region]))
    selected = inside - outside  # masks lying completely within the selection

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
            # a real merge happened -> take the single undo snapshot now
            snapshot_for_undo(rec)
            # compact ids to a contiguous range and remap labels to match
            uniq, inv = np.unique(out, return_inverse=True)
            rec["masks"] = inv.reshape(m.shape).astype(m.dtype)
            remap = {int(old): new for new, old in enumerate(uniq)}
            rec["labels"] = {
                remap[k]: v for k, v in rec.get("labels", {}).items() if remap.get(k, 0)
            }

    st.toast(f"Joined {merged_n} masks in {groups}.")


def assign_clicked():
    """Assign class to mask at clicked location."""

    # check if there was a click
    if not ss["class_click"]:
        return

    # get current record
    rec = get_current_rec()

    # map click (viewport) to full-res image coords
    fx, fy = disp_to_full(ss["class_click"]["x"], ss["class_click"]["y"])
    xy = (
        int(min(max(round(fx), 0), rec["W"] - 1)),
        int(min(max(round(fy), 0), rec["H"] - 1)),
    )

    # ignore click from previous run
    if xy == ss["last_class_xy"]:
        return

    # store last click
    ss["last_class_xy"] = xy

    # assign class to mask at clicked location
    x, y = xy
    m = rec.get("masks")

    iid = int(m[y, x])
    if iid == 0:
        return

    snapshot_for_undo(rec)
    # update label for this instance
    cur = ss.get("side_current_class")
    labels = rec.setdefault("labels", {})
    if cur == "No label" or cur is None:
        labels.pop(iid, None)
    else:
        labels[iid] = cur

    ss["class_click"] = False  # prevent reprocessing on rerun


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
        value=int(ss.get("cp_diameter", 0)),
        step=1,
        help="Leave as 0 for Cellpose to estimate diameter, or set a manual value.",
        key="w_cp_diameter",
    )
    ss["cp_diameter"] = diam_val

    # cellprob threshold
    cellprob = st.number_input(
        "Cell probability threshold",
        value=float(ss.get("cp_cellprob_threshold")),
        step=0.1,
        min_value=-2.0,
        max_value=2.0,
        key="w_cp_cellprob_threshold",
        help="Higher -> fewer cells.",
    )
    ss["cp_cellprob_threshold"] = cellprob

    # Flow threshold
    flowthr = st.number_input(
        "Flow threshold",
        value=float(ss.get("cp_flow_threshold")),
        step=0.1,
        min_value=-2.0,
        max_value=2.0,
        key="w_cp_flow_threshold",
        help="Lower -> more permissive flows.",
    )
    ss["cp_flow_threshold"] = flowthr

    # Minimum size threshold
    min_size = st.number_input(
        "Minimum cell size (pixels)",
        value=int(ss.get("cp_min_size")),
        min_value=0,
        step=10,
        key="w_cp_min_size",
        help="Remove masks smaller than this area.",
    )
    ss["cp_min_size"] = min_size

    # Niter
    niter = st.number_input(
        "Niter",
        value=int(ss["cp_niter"]),
        min_value=0,
        max_value=2000,
        step=10,
        key="w_cp_niter",
        help="Higher values favour longer, stringier, cells.",
    )
    ss["cp_niter"] = niter


def _set_mode(mode: str) -> None:
    """Set the interaction mode via an on_click callback so switching is a single clean
    rerun. An inline st.rerun() instead aborts the run before the zoom panel renders,
    which drops the zoomed view."""
    ss["interaction_mode"] = mode


def render_box_tools_fragment(key_ns="side"):
    """Render SAM2 box drawing and segmentation fragment."""

    # get current record
    rec = get_current_rec()

    # button to set mode to draw boxes on the image
    st.button(
        "Draw box",
        width="stretch",
        key=f"{key_ns}_draw_boxes",
        shortcut="B",
        help="Click and drag boxes around cells (shortcut: B)",
        on_click=_set_mode,
        args=("Draw box",),
    )

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
        st.rerun()


def render_draw_mask_tools_fragment(key_ns="side"):
    """Render the manual mask drawing mode options (Freehand / Ellipse)."""

    c1, c2 = st.columns([1, 1])

    # button to set mode to freehand (lasso) mask drawing
    c1.button(
        "Freehand",
        width="stretch",
        key=f"{key_ns}_draw_masks",
        shortcut="F",
        help="Click and hold to draw a freehand mask (shortcut: F)",
        on_click=_set_mode,
        args=("Freehand",),
    )

    # button to set mode to ellipse mask drawing
    c2.button(
        "Ellipse",
        width="stretch",
        key=f"{key_ns}_draw_ellipse",
        shortcut="E",
        help="Drag a box around a colony to fill it with a rough ellipse (shortcut: E)",
        on_click=_set_mode,
        args=("Ellipse",),
    )

    c3, c4 = st.columns([1, 1])

    # button to set mode to cutting masks with a drawn line
    c3.button(
        "Split masks",
        width="stretch",
        key=f"{key_ns}_cut_mask",
        shortcut="S",
        help="Click and drag a line all the way through a mask to split it in two (shortcut: S)",
        on_click=_set_mode,
        args=("Split masks",),
    )

    # button to set mode to merging two touching masks
    c4.button(
        "Join masks",
        width="stretch",
        key=f"{key_ns}_join_masks",
        shortcut="J",
        help="Draw a lasso around touching masks to join them into one (shortcut: J)",
        on_click=_set_mode,
        args=("Join masks",),
    )


def render_common_tools_fragment(key_ns="tools"):
    """Render the always-visible editing tools (Remove, Clear All, Undo), shown
    outside the segmentation/classification tabs so they're available in both."""

    # get current record
    rec = get_current_rec()

    # button to set mode to remove masks or boxes by clicking on them
    st.button(
        "Remove",
        width="stretch",
        key=f"{key_ns}_remove",
        shortcut="D",
        help="Click masks or boxes to remove them (shortcut: D)",
        on_click=_set_mode,
        args=("Remove",),
    )

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
        st.rerun()

    # single-level undo of the last action
    render_undo_button(c2, key_ns=key_ns)


def _free_arrow_keys_from_slider(slider_key: str) -> None:
    """Blur the zoom slider on focus so Left/Right always change image.

    Streamlit's button shortcuts stand down only for text inputs, but a slider thumb
    handles the arrow keys itself and swallows them. Mouse use is unaffected. The
    listener is delegated so it survives reruns; the window flag stops duplicates."""
    components.html(
        """<script>
const w = window.parent;
if (!w.__mycolArrowNav) {
    w.__mycolArrowNav = true;
    w.document.addEventListener("focusin", (e) => {
        const t = e.target;  // deferred: a synchronous blur here can re-enter
        if (t && t.closest && t.closest("%s")) setTimeout(() => t.blur(), 0);
    });
}
</script>"""
        % f".st-key-{slider_key}",
        height=0,
    )


def render_zoom_controls(key_ns: str = "zoom") -> None:
    """Zoom slider + directional nudge pad, rendered in its own panel beside the image.

    Controlled slider: zoom is stored in a plain session key so it stays in sync with
    the display. This panel must render before the image so the crop reads the new
    zoom on the same run (no one-rerun lag)."""
    rec = get_current_rec()
    if rec is None:
        return

    def _zoom_changed(slider_key: str) -> None:
        ss["zoom"] = float(ss[slider_key])
        ss["_refocus_after_zoom"] = True

    slider_key = f"{key_ns}_slider"
    zoom_value = min(max(float(ss.get("zoom", 1.0)), 1.0), 10.0)
    ss["zoom"] = zoom_value
    ss[slider_key] = zoom_value

    st.slider(
        "Zoom",
        min_value=1.0,
        max_value=10.0,
        step=0.5,
        key=slider_key,
        on_change=_zoom_changed,
        args=(slider_key,),
        help="Zoom into the image; click the minimap or use the arrows to move the view.",
    )
    _free_arrow_keys_from_slider(slider_key)
    _render_minimap(rec)
    _render_nudge_pad(rec, key_ns)
    if ss.pop("_refocus_after_zoom", False):
        _refocus_main_document()


def _render_minimap(rec: Record) -> None:
    """Overview thumbnail of the whole image with a rectangle marking the visible crop.

    Click anywhere on it to recentre the zoomed view there."""
    W, H = rec["W"], rec["H"]
    zoom = max(1.0, float(ss.get("zoom", 1.0)))

    thumb_w = 200
    thumb_h = max(1, round(H * thumb_w / W))
    thumb = (
        Image.fromarray(rec["image"]).resize((thumb_w, thumb_h), Image.BILINEAR).convert("RGB")
    )

    # current crop rectangle, in thumbnail coordinates
    cw, ch = W / zoom, H / zoom
    cx = min(max(float(ss.get("pan_cx", W / 2)), cw / 2), W - cw / 2)
    cy = min(max(float(ss.get("pan_cy", H / 2)), ch / 2), H - ch / 2)
    sx, sy = thumb_w / W, thumb_h / H
    ImageDraw.Draw(thumb).rectangle(
        [(cx - cw / 2) * sx, (cy - ch / 2) * sy, (cx + cw / 2) * sx, (cy + ch / 2) * sy],
        outline=(255, 0, 0),
        width=2,
    )

    streamlit_image_coordinates(
        np.array(thumb),
        key="minimap_click",
        use_column_width="always",
        on_click=_minimap_clicked,
    )


def _minimap_clicked():
    """Recentre the zoomed view on the point clicked in the minimap."""
    click = ss.get("minimap_click")
    if not click or click.get("unix_time") == ss.get("_minimap_last_t"):
        return
    ss["_minimap_last_t"] = click.get("unix_time")
    rec = get_current_rec()
    if rec is None:
        return
    rw = float(click.get("width") or 1)
    rh = float(click.get("height") or 1)
    ss["pan_cx"] = click["x"] * rec["W"] / rw
    ss["pan_cy"] = click["y"] * rec["H"] / rh
    ss["_refocus_after_zoom"] = True


def _render_nudge_pad(rec: Record, key_ns: str = "tools") -> None:
    """Directional d-pad that shifts the zoomed view by ~a quarter of the visible crop.

    Disabled at zoom 1, where the whole image is already shown."""
    W, H = rec["W"], rec["H"]
    zoom = max(1.0, float(ss.get("zoom", 1.0)))
    cw, ch = W / zoom, H / zoom
    step_x, step_y = 0.25 * cw, 0.25 * ch
    disabled = zoom <= 1.0

    def nudge(dx, dy):
        # clamp to the valid centre range so repeated nudges don't overshoot the edge
        cx = float(ss.get("pan_cx", W / 2)) + dx * step_x
        cy = float(ss.get("pan_cy", H / 2)) + dy * step_y
        ss["pan_cx"] = min(max(cx, cw / 2), W - cw / 2)
        ss["pan_cy"] = min(max(cy, ch / 2), H - ch / 2)

    def centre():
        ss["pan_cx"], ss["pan_cy"] = W / 2, H / 2

    _, up, _ = st.columns(3)
    up.button("▲", key=f"{key_ns}_pan_up", width="stretch", disabled=disabled,
              on_click=nudge, args=(0, -1), help="Move view up")
    left, mid, right = st.columns(3)
    left.button("◀", key=f"{key_ns}_pan_left", width="stretch", disabled=disabled,
                on_click=nudge, args=(-1, 0), help="Move view left")
    mid.button("◎", key=f"{key_ns}_pan_centre", width="stretch", disabled=disabled,
               on_click=centre, help="Centre the view")
    right.button("▶", key=f"{key_ns}_pan_right", width="stretch", disabled=disabled,
                 on_click=nudge, args=(1, 0), help="Move view right")
    _, down, _ = st.columns(3)
    down.button("▼", key=f"{key_ns}_pan_down", width="stretch", disabled=disabled,
                on_click=nudge, args=(0, 1), help="Move view down")


def _undo_clicked():
    """Undo callback (on_click so it's a single clean rerun that keeps the zoom view)."""
    apply_undo(get_current_rec())


def render_undo_button(container=st, key_ns="side"):
    """Render the undo button, reverting the last mask/box action via apply_undo
    (single-level; only the most recent action is recoverable)."""
    container.button(
        "Undo",
        width="stretch",
        key=f"{key_ns}_undo",
        shortcut="ctrl+z",
        help="Undo the last action — only the most recent action can be undone (shortcut: Ctrl+Z)",
        on_click=_undo_clicked,
    )


# -----------------------------------------------------#
# ---------------- RENDER MAIN DISPLAY --------------- #
# -----------------------------------------------------#


def _normalized_display_image(image: ImageArray) -> ImageArray:
    """normalize_image(image), memoised in one session slot keyed on image identity.

    The display path only ever normalises the current image, so a single slot serves
    every repeat rerun on it; memory stays bounded to one normalized copy regardless
    of how many images are viewed."""
    slot = ss.get("_norm_display_slot")
    if slot is not None and slot[0] is image:
        return slot[1]
    norm = normalize_image(image)
    ss["_norm_display_slot"] = (image, norm)
    return norm


def render_display_and_interact_fragment(key_ns="edit"):
    """Render main image display and interaction.

    Not a fragment: the zoom/pan controls live in the separate tools panel, and a
    fragment here would render from a stale session-state scope (zoom would desync
    from the slider). Full reruns keep the view and the image in sync."""

    # get current record and verify that images are uploaded
    rec = get_current_rec()

    # display image with masks overlay and interaction (zoom/pan are driven from the
    # tools panel; the view lives in session state so it persists across mode switches)
    rec_for_disp = rec
    if ss.get("show_normalized"):  # normalize background image if selected
        rec_for_disp = {**rec, "image": _normalized_display_image(rec["image"])}

    base_img, disp_w, disp_h = create_image_display(rec_for_disp)

    # handle interaction modes for the image (e.g. draw box, draw mask, remove mask, etc)
    mode = ss.get("interaction_mode", "Draw box")  # default to draw box
    if mode == "Pan":  # Pan mode was removed in favour of the nudge d-pad
        ss["interaction_mode"] = mode = "Draw box"
    if mode == "Freehand":
        _handle_draw_mask_mode(
            rec=rec,
            base_img=base_img,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Ellipse":
        _handle_draw_ellipse_mode(
            rec=rec,
            base_img=base_img,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Split masks":
        _handle_cut_mask_mode(
            rec=rec,
            base_img=base_img,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Draw box":
        _handle_draw_box_mode(
            rec=rec,
            base_img=base_img,
            disp_w=disp_w,
            disp_h=disp_h,
            key_ns=key_ns,
        )
    elif mode == "Join masks":
        _handle_join_mask_mode(
            rec=rec,
            base_img=base_img,
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
