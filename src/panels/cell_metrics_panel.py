# panels/cell_metrics_panel.py
import streamlit as st

from src.helpers.cell_metrics_functions import (
    build_analysis_df,
    plot_violin,
    plot_bar,
)
from src.helpers.grid_helpers import render_image_selection_table
from src.helpers.help_panels import shape_metric_help


def ui_image_selection_container() -> list[str]:
    """Render the shared image-selection table; return selected image names."""
    selected_keys = render_image_selection_table("cell_metrics")
    return [st.session_state["images"][k]["name"] for k in selected_keys]


@st.fragment
def render_plotting_options():
    # ---- Step 1: select images to compare ----
    with st.container(border=True):
        st.subheader("Step 1: Select images to compare")
        selected_names = ui_image_selection_container()
        st.session_state["cell_metrics_selected_names"] = selected_names

    # build the analysis dataframe for the selected images
    df = build_analysis_df(st.session_state["images"])
    if selected_names is not None:
        df = df[df["image"].isin(selected_names)]

    # ---- Step 2: select classes to compare ----
    with st.container(border=True):
        st.subheader("Step 2: Select classes to compare")
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

    # ---- Step 3: select attributes to compare ----
    with st.container(border=True):
        st.subheader("Step 3: Select attributes to compare")
        metric_options = [
            col
            for col in df.columns
            if col not in ["image #", "image", "mask #", "mask label"]
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
            default=st.session_state.get("analysis_plot_type", "Violin"),
            selection_mode="single",
            width="stretch",
        )
        if plot_type:
            st.session_state.analysis_plot_type = plot_type

        # overlay + unit-conversion options as pills in one row
        OVERLAY, CONVERT = "Overlay datapoints", "Convert to distance"
        default_opts = []
        if st.session_state.get("overlay_datapoints", False):
            default_opts.append(OVERLAY)
        if st.session_state.get("convert_to_distance", False):
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
        st.session_state["overlay_datapoints"] = OVERLAY in opts
        st.session_state["convert_to_distance"] = CONVERT in opts

        # pixel size lives in the reserved column, shown only when converting
        if st.session_state["convert_to_distance"]:
            st.session_state["pixel_size"] = col_px.number_input(
                "Pixel size",
                min_value=0.0,
                value=st.session_state.get("pixel_size", 1.0),
                key="pixel_size_input",
                help="Physical size of one pixel.",
            )

    # ---- Step 5: generate and review the plots ----
    with st.container(border=True):
        st.subheader("Step 5: Generate plots")
        if st.button("Generate Plots", width="stretch", type="primary"):
            render_plotting_main()


def render_plotting_main():

    # build dataframes
    df = build_analysis_df(st.session_state["images"])
    selected_names = st.session_state.get("cell_metrics_selected_names")
    if selected_names is not None:
        df = df[df["image"].isin(selected_names)]

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
