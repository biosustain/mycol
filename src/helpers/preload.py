import streamlit as st


def eager_load_heavy_libs():
    """
    Eagerly import heavy libraries to avoid lag when switching tabs
    --> moves the loading time to the inital app startup
    """

    with st.spinner("Loading AI libraries (Torch, Cellpose, SAM2)..."):



        # IO and Components

        try:
            import streamlit_image_coordinates
        except ImportError:
            pass


        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            pass

