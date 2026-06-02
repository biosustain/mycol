import streamlit as st
from src.panels import fine_tune_panel
from src.helpers.state_ops import ordered_keys, require_images
import numpy as np

require_images()

# warning if no images have masks
if not any(np.any(st.session_state["images"][k]["masks"]) for k in ordered_keys()):
    st.warning("⚠️ Please upload or create masks for at least one image.")
    st.stop()

with st.spinner("Loading Training Panel..."):
    col1, col2 = st.columns([1, 1])

    # cellpose_tab, densenet_tab = st.tabs(
    #     [
    #         "Train a Cellpose model to identify cells",
    #         "Train a Densenet model to classify cells",
    #     ]
    # )

    training_tab = st.segmented_control(
        "",
        options=[
            "Train a Cellpose model to identify cells",
            "Train a DenseNet model to classify cells",
        ],
        default="Train a Cellpose model to identify cells",
        selection_mode="single",
        width="stretch",
        key="training_options",
    )

    if training_tab == "Train a Cellpose model to identify cells":
        fine_tune_panel.render_cellpose_options()
        fine_tune_panel.render_cellpose_train_fragment()
        fine_tune_panel.show_cellpose_training_plots()

    else:
        fine_tune_panel.render_densenet_options()
        fine_tune_panel.render_densenet_train_fragment()
        fine_tune_panel.show_densenet_training_plots()
