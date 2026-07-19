# panels/cell_metrics_panel.py
import streamlit as st

from src.helpers.cell_metrics_functions import (
    build_analysis_df,
    available_labels,
    METRIC_COLS,
    plot_violin,
    plot_bar,
)
from src.helpers.grid_helpers import render_image_selection_table
from src.helpers.help_panels import shape_metric_help

ss = st.session_state


@st.fragment
def render_plotting_options():
    # ---- Step 1: select images to compare ----
    with st.container(border=True):
        st.subheader("Step 1: Select images to compare")
        selected_keys = render_image_selection_table("cell_metrics")
        selected_names = [ss["images"][k]["name"] for k in selected_keys]
        ss["cell_metrics_selected_names"] = selected_names

    # ---- Step 2: select classes to compare ----
    with st.container(border=True):
        st.subheader("Step 2: Select classes to compare")
        # labels come from the tiny per-image label dicts, so pickers never rebuild the DataFrame
        label_options = available_labels(selected_keys)
        default_labels = ss.get("analysis_labels", label_options)
        default_labels = [
            label for label in default_labels if label in label_options
        ] or label_options
        st.multiselect(
            "Choose classes to compare",
            options=label_options,
            default=default_labels,
            key="analysis_labels",
        )

    # ---- Step 3: select attributes to compare ----
    with st.container(border=True):
        st.subheader("Step 3: Select attributes to compare")
        metric_options = METRIC_COLS
        default_metrics = ss.get("analysis_metrics", ["area"])
        default_metrics = [
            m for m in default_metrics if m in metric_options
        ] or ["area"]

        st.segmented_control(
            "Choose cell descriptors to compare",
            options=metric_options,
            default=default_metrics,
            selection_mode="multi",
            key="analysis_metrics",
            width="stretch",
        )

        with st.popover(label="Attribute Information"):
            shape_metric_help()

    # ---- Step 4: select plot parameters ----
    with st.container(border=True):
        st.subheader("Step 4: Select plot parameters")
        # third column is reserved for the pixel-size input so the other
        # controls don't shift when it appears
        col_plot, col_opts, col_px = st.columns([2, 3, 2])

        # choose plot type
        plot_type = col_plot.pills(
            "Plot type",
            ["Violin", "Bar"],
            default=ss.get("analysis_plot_type", "Violin"),
            selection_mode="single",
            width="stretch",
        )
        if plot_type:
            ss.analysis_plot_type = plot_type

        # overlay + unit-conversion options as pills in one row
        OVERLAY, CONVERT = "Overlay datapoints", "Convert to distance"
        default_opts = []
        if ss.get("overlay_datapoints", False):
            default_opts.append(OVERLAY)
        if ss.get("convert_to_distance", False):
            default_opts.append(CONVERT)
        opts = (
            col_opts.pills(
                "Options",
                [OVERLAY, CONVERT],
                default=default_opts,
                selection_mode="multi",
                width="stretch",
            )
            or []
        )
        ss["overlay_datapoints"] = OVERLAY in opts
        ss["convert_to_distance"] = CONVERT in opts

        # pixel size lives in the reserved column, shown only when converting
        if ss["convert_to_distance"]:
            ss["pixel_size"] = col_px.number_input(
                "Pixel size",
                min_value=0.0,
                value=ss.get("pixel_size", 1.0),
                key="pixel_size_input",
                help="Physical size of one pixel.",
            )

    # ---- Step 5: generate and review the plots ----
    with st.container(border=True):
        st.subheader("Step 5: Generate plots")
        if st.button("Generate Plots", width="stretch", type="primary"):
            render_plotting_main()


def render_plotting_main():

    # build dataframes (only runs here, on the Generate Plots button click)
    df = build_analysis_df(ss["images"])
    selected_names = ss.get("cell_metrics_selected_names")
    if selected_names is not None:
        df = df[df["image"].isin(selected_names)]

    df_filt = df.copy()
    df_filt["mask label"] = df_filt["mask label"].fillna("Unlabelled")

    # filter by selected labels
    labels_to_plot = ss.get("analysis_labels", None)
    metrics = ss.get("analysis_metrics") or [
        "area",
        "perimeter",
    ]

    if labels_to_plot:
        df_filt = df_filt[df_filt["mask label"].isin(labels_to_plot)]

    if df_filt.empty:
        st.info("No data for the selected labels.")
        return

    # plot each metric
    ptype = ss.get("analysis_plot_type", "Violin")
    for col in metrics:
        fname, fig = (plot_violin if ptype == "Violin" else plot_bar)(df_filt, col)
        title = "Cell " + col.replace("_", " ").title()
        st.header(title)
        st.plotly_chart(fig, width="stretch")
        ss[fname] = fig
