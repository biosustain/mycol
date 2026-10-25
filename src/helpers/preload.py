import streamlit as st


def eager_load_heavy_libs():
    """
    Eagerly import heavy libraries to avoid lag when switching tabs
    --> moves the loading time to the inital app startup
    """

    with st.spinner("Loading MobileSAM and image tools..."):

        try:
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError:
            pass
