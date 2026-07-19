import streamlit as st

ss = st.session_state


def option_row(label, default, key, caption, disabled=False, on_change=None):
    """Checkbox with an explanatory caption; returns whether it is ticked."""
    # The tick is stored under `key`, which survives page navigation.
    wkey = f"{key}_widget"
    ss.setdefault(wkey, ss.get(key, default))

    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        checked = st.checkbox(label, key=wkey, on_change=on_change, disabled=disabled)
    with c2:
        st.caption(caption)

    ss[key] = checked
    return checked
