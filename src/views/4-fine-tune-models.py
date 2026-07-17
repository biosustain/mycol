import streamlit as st
from src.panels import fine_tune_panel
from src.helpers.state_ops import ordered_keys, require_images
import numpy as np

require_images()

# warning if no images have masks
if not any(np.any(st.session_state["images"][k]["masks"]) for k in ordered_keys()):
    st.warning("⚠️ Please upload or create masks for at least one image.")
    st.stop()

CELLPOSE = "Train a Cellpose model to identify cells"
DENSENET = "Train a DenseNet model to classify cells"

with st.spinner("Loading Training Panel..."):

    # ---- Step 1: choose what to fine-tune ----
    with st.container(border=True):
        st.subheader("Step 1: Choose which model to fine-tune")
        training_tab = st.pills(
            "Model type",
            options=[CELLPOSE, DENSENET],
            default=CELLPOSE,
            selection_mode="single",
            width="stretch",
            key="training_options",
            label_visibility="collapsed",
        )

    # segmented_control returns None when deselected; fall back to the default
    is_cellpose = training_tab != DENSENET
    model_label = "Cellpose segmenter" if is_cellpose else "DenseNet classifier"

    # ---- Step 2: choose the images for the training set ----
    with st.container(border=True):
        st.subheader("Step 2: Select images to train on")
        if is_cellpose:
            fine_tune_panel.render_cellpose_image_selection()
        else:
            fine_tune_panel.render_densenet_image_selection()

    # ---- Step 3: choose the training parameters ----
    with st.container(border=True):
        st.subheader(f"Step 3: Set the {model_label} training parameters")
        if is_cellpose:
            fine_tune_panel.render_cellpose_params()
        else:
            fine_tune_panel.render_densenet_params()

    # ---- Step 4: train the model ----
    with st.container(border=True):
        st.subheader(f"Step 4: Train the {model_label}")
        if is_cellpose:
            fine_tune_panel.render_cellpose_train_fragment()
        else:
            fine_tune_panel.render_densenet_train_fragment()

    # ---- Step 5: review model training (only once a model is trained) ----
    trained = (
        "cellpose_training_losses" in st.session_state
        if is_cellpose
        else "densenet_training_metrics" in st.session_state
    )
    if trained:
        with st.container(border=True):
            st.subheader("Step 5: Review model training")
            if is_cellpose:
                fine_tune_panel.show_cellpose_training_plots()
            else:
                fine_tune_panel.show_densenet_training_plots()
