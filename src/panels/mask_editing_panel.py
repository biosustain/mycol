# panels/edit_masks.py
import streamlit as st

from src.helpers.state_ops import ordered_keys
from src.helpers.mask_editing_functions import (
    render_cellpose_hyperparameters_fragment,
    render_box_tools_fragment,
    render_draw_mask_tools_fragment,
    render_mask_tools_fragment,
    render_display_and_interact_fragment,
)
from src.helpers.classifying_functions import (
    classify_actions_fragment,
    class_selection_fragment,
    class_manage_fragment,
)
from src.helpers.cellpose_functions import (
    segment_current_and_refresh,
    batch_segment_and_refresh,
)

# ---------- Rendering functions ----------


def _mode_display_text() -> str:
    """Return the current mode text, with class name if in Assign class mode."""
    mode = st.session_state["interaction_mode"]
    if mode == "Assign class":
        cls = st.session_state.get("side_current_class", "No label")
        return f"{mode} ({cls})"
    return mode


def render_segment_sidebar(*, key_ns: str = "side"):
    with st.container(border=True):
        st.info(f"Current Mode: *{_mode_display_text()}*")

        # render cellpose controls
        with st.popover(
            "Predict masks with Cellpose",
            width="stretch",
            help="Segment cells using the loaded Cellpose model.",
            type="primary",
        ):

            model_options = ["Cyto3", "Cyto2", "Fine-tuned Model"]
            finetuned_available = st.session_state["cellpose_model_bytes"] is not None
            default_index = (
                model_options.index("Fine-tuned Model") if finetuned_available else 0
            )
            model_family = st.selectbox(
                "Select model", model_options, index=default_index
            )

            model_type_map = {"Cyto3": "cyto3", "Cyto2": "cyto2", "Fine-tuned Model": None}
            model_type = model_type_map[model_family]
            disabled = model_type is None and st.session_state["cellpose_model_bytes"] is None

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Generate",
                    width="stretch",
                    key="segment_image",
                    help="Segment this image with Cellpose.",
                    disabled=disabled,
                ):
                    segment_current_and_refresh(model_type)

            with col2:
                if st.button(
                    "Batch generate",
                    width="stretch",
                    key="batch_segment_image",
                    help="Segment all uploaded images with Cellpose.",
                    disabled=disabled,
                ):
                    batch_segment_and_refresh(model_type)

            st.caption("Change hyperparameters to increase accuracy:")

            with st.expander(
                "Cellpose hyperparameters",
            ):
                render_cellpose_hyperparameters_fragment()

        # render SAM2 controls
        with st.popover(
            "Predict masks with boxes",
            width="stretch",
            help="Draw boxes and click segment to use SAM2 to segment individual cells.",
            type="primary",
        ):
            render_box_tools_fragment(key_ns)

        # render manual mask drawing controls (Freehand / Ellipse)
        with st.popover(
            "Manually draw masks",
            width="stretch",
            help="Manually draw masks freehand or as ellipses.",
            type="primary",
        ):
            render_draw_mask_tools_fragment(key_ns)

        # section for selecting tools for directly adding/removing masks
        render_mask_tools_fragment(key_ns)


def render_classify_sidebar(*, key_ns: str = "side"):

    with st.container(border=True):
        st.info(f"Current Mode: *{_mode_display_text()}*")

        with st.popover(label="Manage Labels", width="stretch", type="primary"):
            class_manage_fragment(key_ns)  # add/delete/rename

        # Action buttons to classify cells with Densenet
        with st.popover(
            "Classify cells with Densenet", width="stretch", type="primary"
        ):

            classify_actions_fragment()

        class_selection_fragment()


def render_main(*, key_ns: str = "edit"):

    render_display_and_interact_fragment(key_ns=key_ns, max_display_width=768)
