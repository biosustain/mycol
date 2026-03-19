# panels/cell_metrics_panel.py
import streamlit as st

from src.helpers.cell_metrics_functions import (
    build_analysis_df,
    plot_violin,
    plot_bar,
    build_cell_metrics_csv,
)
from src.helpers.help_panels import shape_metric_help


@st.fragment
def render_plotting_options():
    col1, col2 = st.columns([2, 5])
    with col1:
        inner_col1, inner_col2 = st.columns(2, vertical_alignment="center")
        # choose plot type
        plot_type = inner_col1.pills(
            "Plot type",
            ["Violin", "Bar"],
            default=st.session_state.get("analysis_plot_type", "Violin"),
            selection_mode="single",
            width="stretch",
        )
        if plot_type:
            st.session_state.analysis_plot_type = plot_type

        # toggle overlay of datapoints in the plots
        overlay_points = inner_col2.toggle(
            "Overlay datapoints",
            value=st.session_state.get("overlay_datapoints", False),
            key="overlay_datapoints",
        )

        with st.popover(label="Descriptor Information", width="stretch"):
            shape_metric_help()

        # pixel-to-distance conversion
        chk_col, px_col = st.columns([2, 2], vertical_alignment="center")
        chk_col.checkbox(
            "Convert pixels to distance",
            value=st.session_state.get("convert_to_distance", False),
            key="convert_to_distance",
            help="Apply pixel size to convert measurements to physical units.",
        )
        if st.session_state.get("convert_to_distance"):
            pixel_size = px_col.number_input(
                "Pixel size",
                min_value=0.0,
                value=st.session_state.get("pixel_size", 1.0),
                key="pixel_size_input",
                help="Physical size of one pixel.",
            )
            st.session_state["pixel_size"] = pixel_size

    with col2:
        # build the analysis dataframe
        df = build_analysis_df(st.session_state["images"])
        if df.empty:
            st.info("No masks found.")
            return

        # Labels multiselect (single instance)
        label_options = sorted(
            df["mask label"].unique(), key=lambda x: (x != "Unlabelled", str(x))
        )
        default_labels = st.session_state.get("analysis_labels", label_options)
        default_labels = [
            label for label in default_labels if label in label_options
        ] or label_options
        st.multiselect(
            "Choose classes to compare",
            options=label_options,
            default=default_labels,
            key="analysis_labels",
        )

        # Metrics multiselect (single instance)
        metric_options = [
            col
            for col in df.columns
            if col not in ["image", "mask #", "mask label"]
        ]
        default_metrics = st.session_state.get("analysis_metrics", metric_options)
        default_metrics = [
            m for m in default_metrics if m in metric_options
        ] or metric_options

        st.segmented_control(
            "Choose cell descriptors to compare",
            options=metric_options,
            default="area",
            selection_mode="multi",
            key="analysis_metrics",
            width="stretch",
        )

    btn_col1, btn_col2 = st.columns(2)
    if btn_col1.button("Generate Plots", width="stretch", type="primary"):
        render_plotting_main()

    btn_col2.download_button(
        "Download table of cell descriptors",
        data=build_cell_metrics_csv(
            tuple(st.session_state.get("analysis_labels") or ())
        ),
        file_name="cell_metrics.csv",
        mime="text/csv",
        width="stretch",
        key="dl_cell_metrics_csv",
        type="primary",
    )


def render_plotting_main():

    # build dataframes
    df = build_analysis_df(st.session_state["images"])

    df_filt = df.copy()
    df_filt["mask label"] = (
        df_filt["mask label"].replace("Remove label", None).fillna("Unlabelled")
    )

    # filter by selected labels
    labels_to_plot = st.session_state.get("analysis_labels", None)
    metrics = st.session_state.get("analysis_metrics") or [
        "area",
        "perimeter",
    ]

    if labels_to_plot:
        df_filt = df_filt[df_filt["mask label"].isin(labels_to_plot)]

    if df_filt.empty:
        st.info("No data for the selected labels.")
        return

    # plot each metric
    ptype = st.session_state.get("analysis_plot_type", "Violin")
    for col in metrics:
        fname, fig = (plot_violin if ptype == "Violin" else plot_bar)(df_filt, col)
        title = "Cell " + col.replace("_", " ").title()
        st.header(title)
        st.plotly_chart(fig, width="stretch")
        st.session_state[fname] = fig
