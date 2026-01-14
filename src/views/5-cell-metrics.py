import streamlit as st
from src.panels import cell_metrics_panel
from src.helpers.state_ops import ordered_keys
import numpy as np

# Warning if no images have been uploaded yet
if st.session_state["images"] == {}:
    st.warning("⚠️ Please upload an image on the 'Upload Models and Data' tab first.")
    st.stop()

# warning if no images have masks
if not any(np.any(st.session_state["images"][k]["masks"]) for k in ordered_keys()):
    st.warning("⚠️ Please upload or create masks for at least one image.")
    st.stop()


with st.spinner("Loading Metrics..."):
    with st.container(border=True):
        cell_metrics_panel.render_plotting_options()
