import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.helpers.state_ops import ordered_keys
from skimage.measure import regionprops
from pathlib import Path
from src.helpers.classifying_functions import color_hex_for


def hex_for_plot_label(label: str) -> str:
    """
    Map plotting labels to the same hex used for masks.
    'Unlabelled' and 'No label' both use the reserved 'No label' color.
    """
    if label in (None, "", "Unlabelled", "No label"):
        return color_hex_for("No label")
    return color_hex_for(label)


def plot_violin(df: pd.DataFrame, value_col: str):
    """
    Create a violin plot of `value_col` grouped by mask label.
    Shows data points if `overlay_datapoints` is set in session state."""
    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))

    # use mask colors
    color_map = {lab: hex_for_plot_label(lab) for lab in order}

    show_points = bool(st.session_state.get("overlay_datapoints", False))
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
                box_visible=True,
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
            imgs = df.loc[idx, "image"].astype(str).to_numpy()
            masks = df.loc[idx, "mask #"].astype(str).to_numpy()
            texts = [f"{im}_patch{mk}" for im, mk in zip(imgs, masks)]

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
                    text=texts,  # per-point labels
                    hovertemplate="Image: %{text}<extra></extra>",
                    hoveron="points",
                    showlegend=False,
                )
            )

    # final layout
    fig.update_layout(
        violinmode="overlay",
        xaxis_title="Label",
        yaxis_title=value_col.replace("_", " ").title(),
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

    df["label"] = df["mask label"].replace("No label", None).fillna("Unlabelled")
    order = sorted(df["label"].unique(), key=lambda x: (x != "Unlabelled", str(x)))
    title_y = value_col.replace("_", " ").title()

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
        hovertemplate="<b>%{x}</b><br>" + title_y + ": %{y:.2f}<extra></extra>",
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
    for i, lab in enumerate(order):
        idx = sub["label"] == lab
        ys = sub.loc[idx, value_col].to_numpy()
        if ys.size == 0:
            continue

        imgs = sub.loc[idx, "image"].astype(str).to_numpy()
        masks = sub.loc[idx, "mask #"].astype(str).to_numpy()
        texts = [f"{Path(im).stem}_patch{mk}" for im, mk in zip(imgs, masks)]

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
                text=texts,
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
    Compute a set of shape metrics for a single skimage regionprops object.

    Parameters
    ----------
    prop : skimage.measure._regionprops.RegionProperties
        Region properties for a single labeled instance.
    max_edge_to_edge : float or None
        Optional precomputed longest internal chord length.
        If provided, will be used for a normalized metric.

    Returns
    -------
    dict
        Keys are metric names (strings), values are floats (or np.nan).
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

    # Aspect ratio (elongated shapes >> 1)
    aspect_ratio = safe_div(major, minor) if minor > 0 else float("nan")

    # Elongation in [0, 1):
    #   0 -> circle-like, ->1 very elongated
    elongation = (
        safe_div(major - minor, major + minor) if (major + minor) > 0 else float("nan")
    )

    # Solidity = area / convex_area (1 for convex shapes)
    solidity = float(getattr(prop, "solidity", float("nan")))

    # Extent = area / bounding_box_area
    extent = float(getattr(prop, "extent", float("nan")))

    # Eccentricity of the ellipse that has the same second-moments
    eccentricity = float(getattr(prop, "eccentricity", float("nan")))

    # Compactness (inverse of circularity, higher = less compact)
    compactness = safe_div(perimeter**2, 4.0 * np.pi * area)

    return {
        "area": area,
        "perimeter": perimeter,
        "major axis length": major,
        "minor axis length": minor,
        "circularity": circularity,
        "roundness": roundness,
        "aspect ratio": aspect_ratio,
        "elongation": elongation,
        "solidity": solidity,
        "extent": extent,
        "eccentricity": eccentricity,
        "compactness": compactness,
    }


@st.cache_data(show_spinner="Building analysis DataFrame...")
def build_analysis_df(records):
    """
    Build a DataFrame with per-mask metrics for all images in session state.

    Columns (existing):
        image, mask #, mask label, mask area, mask perimeter, max edge-to-edge

    New columns (from mask_shape_metrics):
        circularity, roundness, aspect ratio, elongation,
        solidity, extent, eccentricity, compactness, chord / major axis
    """

    rows = []
    # iterate through the image records
    for k in ordered_keys():
        rec = st.session_state.images[k]
        inst = rec.get("masks")
        # skip invalid masks
        if not inst.any():
            continue

        labdict = rec.get("labels", {})  # dict {instance_id -> class/None}
        for prop in regionprops(inst):  # prop.label is the instance id
            iid = int(prop.label)
            cls = labdict.get(iid)

            # compute shape metrics
            shape_metrics = mask_shape_metrics(prop)

            row = {
                "image": rec["name"],
                "mask #": iid,
                "mask label": ("Unlabelled" if cls in (None, "No label") else cls),
            }

            # merge in the shape metrics
            row.update(shape_metrics)

            rows.append(row)

    return pd.DataFrame(rows)


# --- FUNCTIONS FOR DOWNLOADING CLASS CHARACTERISTICS PLOTS


def build_cell_metrics_csv(labels_selected):
    df = build_analysis_df(st.session_state["images"])

    if labels_selected:
        df = df[df["mask label"].isin(labels_selected)]

    if df.empty:
        return ""

    return df.to_csv(index=False)
