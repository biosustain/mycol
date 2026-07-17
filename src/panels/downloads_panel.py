import streamlit as st


def option_row(label, default, key, caption, disabled=False, on_change=None):
    """Render a checkbox with an explanatory caption; return whether it is ticked.

    The tick is mirrored in the plain key ``key``, which is what the zip builders
    read and what survives page navigation; Streamlit drops the widget's own state
    on any rerun where the widget isn't rendered. Seeding the widget key only when
    it is absent restores the tick on returning to the page.
    """
    wkey = f"{key}_widget"
    st.session_state.setdefault(wkey, st.session_state.get(key, default))

    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        checked = st.checkbox(label, key=wkey, on_change=on_change, disabled=disabled)
    with c2:
        st.caption(caption)

    st.session_state[key] = checked
    return checked
