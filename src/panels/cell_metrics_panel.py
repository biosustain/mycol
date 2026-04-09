# panels/cell_metrics_panel.py
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from src.helpers.cell_metrics_functions import (
    build_analysis_df,
    plot_violin,
    plot_bar,
)
from src.helpers.help_panels import shape_metric_help
from src.helpers.state_ops import ordered_keys


def ui_image_selection_container() -> list[str]:
    """Render an AgGrid table for selecting which images to include in plots.
    Returns a list of selected image names."""
    keys = ordered_keys()

    # Collect all label groups across all images
    all_labels: set[str] = set()
    for k in keys:
        rec = st.session_state["images"][k]
        for cls in (rec.get("labels") or {}).values():
            all_labels.add("Unlabelled" if cls in (None, "No label") else cls)
    sorted_labels = sorted(all_labels, key=lambda x: (x != "Unlabelled", x))

    rows = []
    for k in keys:
        rec = st.session_state["images"][k]
        masks = rec.get("masks")
        n_masks = int(masks.max()) if masks is not None and np.any(masks) else 0
        label_dict = rec.get("labels") or {}
        counts: dict[str, int] = {lab: 0 for lab in sorted_labels}
        for cls in label_dict.values():
            lab = "Unlabelled" if cls in (None, "No label") else cls
            counts[lab] = counts.get(lab, 0) + 1
        # masks with no entry in label_dict are unlabelled
        counts["Unlabelled"] = counts.get("Unlabelled", 0) + (n_masks - len(label_dict))
        row = {"_key": k, "Image": rec["name"], "Total masks": n_masks}
        row.update(counts)
        rows.append(row)

    opt = pd.DataFrame(rows)
    ids = opt["_key"].tolist()

    sel = st.session_state.setdefault("cell_metrics_image_sel", {})

    value_cols = ["Total masks"] + sorted_labels
    display_df = opt[["Image"] + value_cols + ["_key"]].copy()

    gb = GridOptionsBuilder.from_dataframe(display_df)
    gb.configure_selection("multiple", use_checkbox=True, rowMultiSelectWithClick=True)
    gb.configure_column("_key", hide=True)
    gb.configure_column("Image", headerCheckboxSelection=True, checkboxSelection=True)
    for col in value_cols:
        gb.configure_column(col, editable=False, type=[])
    grid_options = gb.build()

    pre_selected_rows = [idx for idx, k in enumerate(ids) if sel.get(k, True)]
    grid_response = AgGrid(
        display_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        pre_selected_rows=pre_selected_rows,
        fit_columns_on_grid_load=True,
        height=min(400, 55 + 35 * len(ids)),
        width="100%",
        key="cell_metrics_image_grid",
    )

    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None:
        selected_keys = set()
    elif isinstance(selected_rows, pd.DataFrame):
        selected_keys = set(
            selected_rows.get("_key", pd.Series([], dtype=str)).tolist()
        )
    elif isinstance(selected_rows, list):
        if selected_rows and isinstance(selected_rows[0], dict):
            selected_keys = {
                row.get("_key") for row in selected_rows if row.get("_key")
            }
        else:
            selected_keys = {row for row in selected_rows if isinstance(row, str)}
    else:
        selected_keys = set()

    for k in ids:
        sel[k] = k in selected_keys

    return [st.session_state["images"][k]["name"] for k in ids if sel[k]]


@st.fragment
def render_plotting_options():
    with st.container(border=True):
        st.subheader("Select Images for Analysis")
        selected_names = ui_image_selection_container()
        st.session_state["cell_metrics_selected_names"] = selected_names

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
        inner_col2.toggle(
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
        selected_names = st.session_state.get("cell_metrics_selected_names")
        if selected_names is not None:
            df = df[df["image"].isin(selected_names)]

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
            col for col in df.columns if col not in ["image", "mask #", "mask label"]
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
