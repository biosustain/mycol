import streamlit as st


def option_row(label, default, key, caption, disabled=False, on_change=None):
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        st.checkbox(label, default, key=key, on_change=on_change, disabled=disabled)
    with c2:
        st.caption(caption)
