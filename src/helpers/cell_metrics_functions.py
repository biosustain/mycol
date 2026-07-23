import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.helpers.state_ops import ordered_keys
from src.helpers.plot_helpers import point_hover_texts
from skimage.measure import regionprops
from src.helpers.classifying_functions import color_hex_for

ss = st.session_state


def hex_for_plot_label(label: str) -> str:
    """
    Map plotting labels to the same hex used for masks.
    'Unlabelled' and 'No label' both use the reserved 'No label' color.
    """
    if label in (None, "", "Unlabelled", "No label"):
        return color_hex_for("No label")
    return color_hex_for(label)


def _scale_col(df: pd.DataFrame, col: str):
    """
    Return (scaled_series, axis_label) for `col`, converting to physical units when
    'convert_to_distance' and 'pixel_size' are set in session state.
    Dimensionless metrics are returned as-is.
    """
    pixel_size = ss.get("pixel_size")
    vals = df[col].copy()
    label = col.replace("_", " ").title()

    convert = ss.get("convert_to_distance", False)
    if pixel_size and pixel_size > 0 and convert:
        if col in _AREA_COLS:
            vals = vals * (pixel_size**2)
        elif col in _DIST_COLS:
            vals = vals * pixel_size

    return vals, label


def plot_violin(df: pd.DataFrame, value_col: str):
    """
    Create a violin plot of `value_col` grouped by mask label.
    Shows data points if `overlay_datapoints` is set in session state."""
    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    # use mask colors
    color_map = {lab: hex_for_plot_label(lab) for lab in order}

    scaled_col, y_label = _scale_col(df, value_col)
    df = df.copy()
    df[value_col] = scaled_col

    show_points = bool(ss.get("overlay_datapoints", False))
    fig = go.Figure()

    # violin traces per label
    for lab in order:
        idx = df["label"] == lab
        vals = df.loc[idx, value_col]
        x_vals = [lab] * len(vals)

        # violin body with the mask-matched color
        fig.add_trace(
            go.Violin(
                x=x_vals,
                y=vals,
                name=str(lab),
                legendgroup=str(lab),
                box_visible=False,
                meanline_visible=True,
                line_color="black",
                fillcolor=color_map[lab],
                opacity=0.85,
                points=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # optional data points overlaid
        if show_points and len(vals) > 0:
            fig.add_trace(
                go.Violin(
                    x=x_vals,
                    y=vals,
                    name=f"{lab}_pts",
                    legendgroup=str(lab),
                    box_visible=False,
                    meanline_visible=False,
                    line_color="rgba(0,0,0,0)",
                    fillcolor="rgba(0,0,0,0)",
                    opacity=1.0,
                    points="all",
                    pointpos=0,
                    jitter=0.25,
                    marker=dict(
                        size=4,
                        opacity=0.8,
                        color="black",
                        line=dict(width=0.3, color="black"),
                    ),
                    text=point_hover_texts(
                        df.loc[idx, "image #"],
                        df.loc[idx, "image"],
                        df.loc[idx, "mask #"],
                    ),
                    hovertemplate="%{text}<extra></extra>",
                    hoveron="points",
                    showlegend=False,
                )
            )

    # final layout
    fig.update_layout(
        violinmode="overlay",
        xaxis_title="Label",
        yaxis_title=y_label,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)")

    return f"{value_col.replace(' ', '_')}", fig


def plot_bar(df: pd.DataFrame, value_col: str):
    """
    Create a bar plot of mean `value_col` grouped by mask label,
    with error bars showing standard deviation. Shows data points if `overlay_datapoints` is set
    in session state."""

    df = df.copy()
    scaled_col, title_y = _scale_col(df, value_col)
    df[value_col] = scaled_col

    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    # numeric x positions and colors
    xpos = np.arange(len(order), dtype=float)
    colors = [hex_for_plot_label(lab) for lab in order]

    # bar means + SD
    g = df.groupby("label")[value_col]
    means = g.mean().reindex(order).to_numpy()
    sds = g.std().reindex(order).fillna(0).to_numpy()

    # main bar trace
    bar = go.Bar(
        x=xpos,
        y=means,
        marker_color=colors,
        marker_line=dict(color="black", width=1),
        error_y=dict(type="data", array=sds, visible=True),
        opacity=0.9,
        showlegend=False,
        hovertemplate=f"<b>%{{x}}</b><br>{title_y}: %{{y:.2f}}<extra></extra>",
    )

    # add jittered data points if requested
    traces = add_data_points_to_plot(bar, order, df, value_col, xpos)

    # final layout
    fig = go.Figure(traces, layout=dict(barcornerradius=10))
    fig.update_layout(
        xaxis=dict(
            tickvals=xpos,
            ticktext=order,
            showline=True,
            linecolor="black",
            gridcolor="rgba(0,0,0,0.1)",
        ),
        yaxis=dict(
            title=title_y, showline=True, linecolor="black", gridcolor="rgba(0,0,0,0.1)"
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        bargap=0.3,
        height=500,
        showlegend=False,
    )

    return f"bar_{value_col.replace(' ', '_')}.png", fig


def add_data_points_to_plot(plot, order, sub, value_col, xpos):
    """
    If `overlay_datapoints` is set in session state, add a scatter trace
    with jittered points for each category to the given plotly bar plot."""

    # jittered points per category (optional)
    traces = [plot]
    if not ss.get("overlay_datapoints", False):
        return traces
    for i, lab in enumerate(order):
        idx = sub["label"] == lab
        ys = sub.loc[idx, value_col].to_numpy()
        if ys.size == 0:
            continue

        rng = np.random.default_rng(42 + i)  # deterministic jitter per label
        xj = np.full(ys.shape, xpos[i], dtype=float) + rng.uniform(-0.20, 0.20, ys.size)

        # scatter trace for data points
        traces.append(
            go.Scatter(
                x=xj,
                y=ys,
                mode="markers",
                showlegend=False,
                marker=dict(
                    size=6,
                    opacity=0.75,
                    color="black",
                    line=dict(width=0.3, color="black"),
                ),
                text=point_hover_texts(
                    sub.loc[idx, "image #"],
                    sub.loc[idx, "image"],
                    sub.loc[idx, "mask #"],
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return traces


def safe_div(num, den):
    """Return num / den, but np.nan if den is 0 or not finite."""
    if den == 0 or not np.isfinite(den):
        return float("nan")
    return float(num / den)


def mask_shape_metrics(prop):
    """
    Compute shape metrics for a single skimage regionprops object (one instance).

    Returns {metric_name: float}, with np.nan where a metric is undefined.
    """

    area = float(prop.area)
    perimeter = float(prop.perimeter)

    # Basic axis lengths
    major = float(getattr(prop, "major_axis_length", 0.0))
    minor = float(getattr(prop, "minor_axis_length", 0.0))

    # Classic circularity / roundness measure:
    #   circularity = 4*pi*area / perimeter^2
    # = 1 for a perfect circle, < 1 for other shapes.
    circularity = safe_div(4.0 * np.pi * area, perimeter**2)

    # Alternative "roundness" using major axis:
    #   roundness = 4*area / (pi * major_axis_length^2)
    # again 1 for a perfect circle if major_axis_length is the diameter.
    if major > 0:
        roundness = safe_div(4.0 * area, np.pi * major**2)
    else:
        roundness = float("nan")

    # Solidity = area / convex_area (1 for convex shapes)
    solidity = float(getattr(prop, "solidity", float("nan")))

    # Extent = area / bounding_box_area
    extent = float(getattr(prop, "extent", float("nan")))

    # Eccentricity of the ellipse that has the same second-moments
    eccentricity = float(getattr(prop, "eccentricity", float("nan")))

    # Maximum Feret (caliper) diameter: the longest distance between any two points
    # on the boundary. Unlike major axis length it makes no ellipse assumption.
    # getattr so older skimage builds without the property report nan, not raise.
    feret = float(getattr(prop, "feret_diameter_max", float("nan")))

    return {
        "area": area,
        "perimeter": perimeter,
        "major axis length": major,
        "minor axis length": minor,
        "feret diameter": feret,
        "circularity": circularity,
        "roundness": roundness,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
    }


def white_balance_by_background(
    image: np.ndarray, inst: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """
    White-balance `image` so its background (inst == 0) is neutral gray, making
    cell colour comparable across images with slight lighting differences.
    Referencing the background, not the cells, keeps real cell-colour differences
    intact. Returns float32; unchanged if grayscale or no usable background.
    """
    im = image.astype(np.float32)
    if im.ndim != 3:  # grayscale: nothing to balance
        return im
    bg = inst == 0
    if not bg.any():
        return im
    bg_means = im[bg].mean(axis=0)  # per-channel background mean
    if not np.all(bg_means > eps):
        return im
    im *= bg_means.mean() / bg_means  # flatten each channel's background to gray
    return im


def mask_color_metrics(prop):
    """
    Per-region chromaticity from foreground pixels only. `prop` must carry a
    (white-balanced) intensity_image. Returns brightness-independent channel
    fractions that sum to 1, nan where undefined.
    """
    keys = ("redness", "greenness", "blueness")
    px = prop.image_intensity[prop.image]  # foreground pixels only

    if px.size == 0:
        return dict.fromkeys(keys, float("nan"))
    if px.ndim == 1:  # grayscale: no colour, split evenly
        return dict.fromkeys(keys, 1.0 / 3.0)

    means = px.mean(axis=0)  # per-channel mean
    total = float(means.sum())
    if total <= 0:  # black region: undefined
        return dict.fromkeys(keys, float("nan"))

    return dict(zip(keys, (float(m / total) for m in means[:3])))


# Fixed set of metric columns build_analysis_df emits, so the attribute picker needs no DataFrame.
_DUMMY = regionprops(
    np.ones((3, 3), np.uint8), intensity_image=np.ones((3, 3, 3), np.uint8)
)[0]
METRIC_COLS = list(mask_shape_metrics(_DUMMY).keys()) + list(
    mask_color_metrics(_DUMMY).keys()
)


def available_labels(keys):
    """Distinct class labels across the given images, read cheaply from per-image label dicts."""
    if not keys:
        return []
    classes = set()
    for k in keys:
        for cls in (ss["images"][k].get("labels") or {}).values():
            if cls not in (None, "No label"):
                classes.add(cls)
    return sorted(classes | {"Unlabelled"}, key=lambda x: (x != "Unlabelled", str(x)))


@st.cache_data(show_spinner="Building analysis DataFrame...", max_entries=2)
def build_analysis_df(records):
    """
    Build a DataFrame with per-mask metrics for all images in session state.
    Always returns raw pixel values with original column names.
    """

    rows = []
    # iterate through the image records (img_no matches the 1-based "No." shown
    # in the image selection table)
    for img_no, k in enumerate(ordered_keys(), start=1):
        rec = ss.images[k]
        inst = rec.get("masks")
        # skip invalid masks
        if not inst.any():
            continue

        # per-image resize scale: how many original pixels = 1 stored pixel
        orig_H = rec.get("orig_H", rec["H"])
        orig_W = rec.get("orig_W", rec["W"])
        pixel_scale = max(orig_H, orig_W) / max(rec["H"], rec["W"])

        # white-balance so cell colour is comparable across images
        wb_img = white_balance_by_background(rec["image"], inst)

        labdict = rec.get("labels", {})  # dict {instance_id -> class/None}
        for prop in regionprops(inst, intensity_image=wb_img):  # prop.label is the instance id
            iid = int(prop.label)
            cls = labdict.get(iid)

            # compute shape metrics and normalise to original-pixel units
            shape_metrics = mask_shape_metrics(prop)
            for col in _DIST_COLS:
                if col in shape_metrics:
                    shape_metrics[col] *= pixel_scale
            for col in _AREA_COLS:
                if col in shape_metrics:
                    shape_metrics[col] *= pixel_scale**2

            row = {
                "image #": img_no,
                "image": rec["name"],
                "mask #": iid,
                "mask label": ("Unlabelled" if cls in (None, "No label") else cls),
            }

            # merge in shape + colour metrics
            row.update(shape_metrics)
            row.update(mask_color_metrics(prop))

            rows.append(row)

    return pd.DataFrame(rows)


# Columns that represent distances (multiply by pixel_size).
# Any new distance metric must be listed here, or it is left in stored pixels:
# build_analysis_df uses these sets to undo the upload resize, and _scale_col /
# apply_pixel_scaling use them again to convert to physical units.
_DIST_COLS = {"perimeter", "major axis length", "minor axis length", "feret diameter"}
# Columns that represent areas (multiply by pixel_size²)
_AREA_COLS = {"area"}


def apply_pixel_scaling(df: pd.DataFrame, pixel_size) -> pd.DataFrame:
    """
    Return a copy of df with distance/area columns scaled to physical units.
    Dimensionless metrics (circularity, roundness, etc.) are left unchanged.
    """
    convert = ss.get("convert_to_distance", False)
    if not pixel_size or pixel_size <= 0 or df.empty or not convert:
        return df.drop(columns=["_pixel_scale"], errors="ignore")

    df = df.copy()
    for col in _DIST_COLS:
        if col in df.columns:
            df[col] = df[col] * pixel_size
    for col in _AREA_COLS:
        if col in df.columns:
            df[col] = df[col] * (pixel_size**2)

    return df


def build_per_image_counts(keys=None):
    """Per-image cell counts grouped by class label.

    Single source of truth for per-image counts, shared by the Cell Metrics
    table and the download summary. Returns
    ``(DataFrame, sorted_labels)`` where the frame has columns
    ``_key, Image, Total masks`` followed by one column per label.
    """
    if keys is None:
        keys = ordered_keys()

    # Collect all label groups across all images
    all_labels: set[str] = set()
    for k in keys:
        rec = ss["images"][k]
        for cls in (rec.get("labels") or {}).values():
            all_labels.add("Unlabelled" if cls in (None, "No label") else cls)
    sorted_labels = sorted(all_labels, key=lambda x: (x != "Unlabelled", x))

    rows = []
    for k in keys:
        rec = ss["images"][k]
        masks = rec.get("masks")
        ids = np.unique(masks) if masks is not None else np.zeros(0, dtype=int)
        ids = ids[ids != 0]
        label_dict = rec.get("labels") or {}
        counts: dict[str, int] = {lab: 0 for lab in sorted_labels}
        for iid in ids:
            cls = label_dict.get(int(iid))
            lab = "Unlabelled" if cls in (None, "No label") else cls
            counts[lab] = counts.get(lab, 0) + 1
        row = {"_key": k, "Image": rec["name"], "Total masks": int(ids.size)}
        row.update(counts)
        rows.append(row)

    return pd.DataFrame(rows), sorted_labels


# --- FUNCTIONS FOR DOWNLOADING CLASS CHARACTERISTICS PLOTS


def build_cell_metrics_csv(labels_selected):
    df = build_analysis_df(ss["images"])

    if labels_selected:
        df = df[df["mask label"].isin(labels_selected)]

    if df.empty:
        return ""

    pixel_size = ss.get("pixel_size")
    df = apply_pixel_scaling(df, pixel_size)

    return df.to_csv(index=False)
