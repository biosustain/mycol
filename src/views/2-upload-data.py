from src.panels import upload_panel

import streamlit as st

# Main area
with st.spinner("Loading Upload Panel..."):
    upload_panel.render_main()
