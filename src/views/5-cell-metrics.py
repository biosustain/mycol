import streamlit as st
from src.panels import cell_metrics_panel
from src.helpers.state_ops import ordered_keys, require_images
import numpy as np

require_images()

# warning if no images have masks
if not any(np.any(st.session_state["images"][k]["masks"]) for k in ordered_keys()):
    st.warning("⚠️ Please upload or create masks for at least one image.")
    st.stop()


with st.spinner("Loading Metrics..."):
    cell_metrics_panel.render_plotting_options()
