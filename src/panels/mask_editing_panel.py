# panels/edit_masks.py
import streamlit as st

from src.helpers.state_ops import ordered_keys
from src.helpers.mask_editing_functions import (
    render_cellpose_hyperparameters_fragment,
    render_box_tools_fragment,
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
    segment_current_and_refresh_cellpose_sam,
    batch_segment_current_and_refresh_cellpose_sam,
)

from src.helpers.upload_download_functions import (
    build_masks_images_zip,
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
            "Predict masks for image",
            width="stretch",
            help="Segment cells using the loaded Cellpose model.",
            type="primary",
        ):

            model_family = st.selectbox(
                "Select model", ["Cellpose4", "Fine-tuned Model"]
            )

            col1, col2 = st.columns(2)

            if model_family == "Cellpose4":

                with col1:

                    if st.button(
                        "Generate",
                        width="stretch",
                        key="segment_image_SAM",
                    ):
                        segment_current_and_refresh_cellpose_sam()

                with col2:
                    if st.button(
                        "Batch generate",
                        width="stretch",
                        key="batch_segment_image_sam",
                        help="Segment all uploaded images with Cellpose.",
                    ):
                        batch_segment_current_and_refresh_cellpose_sam()

            else:

                with col1:

                    if st.button(
                        "Generate",
                        width="stretch",
                        key="segment_image",
                        help="Segment this image with Cellpose.",
                        disabled=st.session_state["cellpose_model_bytes"] == None,
                    ):
                        segment_current_and_refresh()
                with col2:
                    if st.button(
                        "Batch generate",
                        width="stretch",
                        key="batch_segment_image",
                        help="Segment all uploaded images with Cellpose.",
                        disabled=st.session_state["cellpose_model_bytes"] == None,
                    ):
                        batch_segment_and_refresh()

            st.caption("Change hyperparameters to increase accuracy:")

            with st.expander(
                "Cellpose hyperparameters",
            ):
                render_cellpose_hyperparameters_fragment()

        # render SAM2 controls
        with st.popover(
            "Predict masks from boxes",
            width="stretch",
            help="Draw boxes and click segment to use SAM2 to segment individual cells.",
            type="primary",
        ):
            render_box_tools_fragment(key_ns)

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

    render_display_and_interact_fragment(key_ns=key_ns, scale=1.5)


def render_download_button():
    if not ordered_keys():
        st.info("Upload data and label masks first.")
        return False

    images = st.session_state.get("images", {})
    ok = ordered_keys() if images else []

    with st.container(border=True):
        with st.popover(label="Download options", width="stretch", type="primary"):
            include_overlay = st.checkbox(
                "Include colored mask overlays", True, key="dl_include_overlay"
            )
            include_counts = st.checkbox(
                "Overlay per-image class counts", False, key="dl_include_counts"
            )
            st.checkbox(
                "Normalize downloaded images", False, key="dl_normalize_download"
            )

            include_patches = st.checkbox(
                "Include cell patch images", False, key="dl_include_patches"
            )

            include_summary = st.checkbox(
                "Include table of per image cell counts",
                True,
                key="dl_include_summary",
            )

            # 🔹 Only build the dataset when the user actually clicks the button
            if st.button(
                "Prepare annotated images for download",
                width="stretch",
                type="primary",
            ):
                mz = build_masks_images_zip(
                    images,
                    ok,
                    include_overlay,
                    include_counts,
                    include_patches,
                    include_summary,
                )
                st.download_button(
                    "Download dataset",
                    mz,
                    "masks_and_images.zip",
                    "application/zip",
                    width="stretch",
                    type="primary",
                )
