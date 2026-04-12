# app.py
import os
import streamlit as st

st.set_page_config(page_title="Mycol", page_icon="👨🏼‍🔬", layout="wide")

st.logo("logo.png", link="https://sambra95.github.io/mycol/index.html")

# Eager load heavy libraries to prevent lag on tab switching
with st.empty():
    st.write("### ⏳ Initializing AI Models...")
    st.caption("Pre-loading PyTorch, Cellpose, and SAM2 for smoother performance.")
    from src.helpers.preload import eager_load_heavy_libs

    eager_load_heavy_libs()


# ------------------ Boot steps ------------------ #
from src.boot import configure_tf_cpu_only
from src.helpers.state_ops import ensure_global_state

ensure_global_state()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
configure_tf_cpu_only()


# restyle top navbar
st.html(
    """
<style>

.stAppHeader { background-color: #E9F2FF; !important; }

.stAppHeader span, .stAppHeader div {
    font-weight: 600 !important;
}

.stAppHeader {
    padding: 12px 20px !important;
}

.stAppHeader {
    box-shadow: 0 2px 20px rgba(0,0,0,0.2) !important;
}

/* Increase text size inside the navbar/header */
.stAppHeader span, .stAppHeader h1, .stAppHeader div {
    font-size: 20px !important;
}
</style>
"""
)

# ------------------ Define pages ------------------ #
pages = [
    st.Page(
        "src/views/2-upload-data.py",
        title="Upload Models and Data",
        default=True,
    ),
    st.Page(
        "src/views/3-create-and-edit-masks.py",
        title="Annotate Images",
    ),
    st.Page(
        "src/views/4-fine-tune-models.py",
        title="Train Models",
    ),
    st.Page(
        "src/views/5-cell-metrics.py",
        title="Visualize Cell Attributes",
    ),
    st.Page(
        "src/views/6-downloads.py",
        title="Downloads",
    ),
]

# ------------------ TOP navigation ------------------ #
nav = st.navigation(pages, position="top", expanded=False)

# ------------------ Run selected page ------------------ #
nav.run()
