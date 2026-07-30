import streamlit as st
from src.panels import mask_editing_panel
from src.helpers.state_ops import (
    ordered_keys,
    set_current_by_index,
    require_images,
    reset_undo_on_navigation,
)

ss = st.session_state

require_images()

with st.spinner("Loading Annotator..."):
    col1, col2, col3 = st.columns([2, 4, 2], gap="large")
    with col1:

        with st.container(border=True, height=770, vertical_alignment="center",):

            ok = ordered_keys()
            names = [ss.images[k]["name"] for k in ok]

            # --- Ensure we have a current key ---
            if (
                "current_key" not in ss
                or ss.current_key not in ok
            ):
                ss.current_key = ok[0]

            reck = ss.current_key
            rec_idx = ok.index(reck) if reck in ok else 0  # 0-based index

            # discard the undo snapshot when the displayed image changes
            reset_undo_on_navigation()

            st.info(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

            # removed slider and buttons if only one image to prevent crash
            if len(ok) != 1:

                # --- Initialize slider state from current image (first run only) ---
                if "slider_jump" not in ss:
                    # slider is 1-based, rec_idx is 0-based
                    ss.slider_jump = rec_idx + 1

                # --- Helper: keep current_key in sync with slider value ---
                def set_current_from_slider():
                    ok_local = ordered_keys()
                    # slider is 1..len(ok), convert to 0-based index, clamp to range
                    idx = max(
                        0, min(len(ok_local) - 1, ss.slider_jump - 1)
                    )
                    set_current_by_index(idx)

                # --- Navigation buttons & slider ---
                nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])

                with nav_col1:
                    if st.button("", width="stretch", shortcut="left"):
                        # move slider one step back, then update current image
                        ss.slider_jump = max(
                            1, ss.slider_jump - 1
                        )
                        set_current_from_slider()
                        st.rerun()

                with nav_col3:
                    if st.button("", width="stretch", shortcut="right"):
                        # move slider one step forward, then update current image
                        ss.slider_jump = min(
                            len(ok), ss.slider_jump + 1
                        )
                        set_current_from_slider()
                        st.rerun()

                with nav_col2:
                    # slider drives the current image via on_change callback
                    st.slider(
                        "Image index",
                        1,
                        len(ok),
                        key="slider_jump",
                        label_visibility="collapsed",
                        on_change=set_current_from_slider,
                    )

            # --- View toggle buttons (active view shows as a filled/primary button) ---
            _views = [
                ("Masks", "show_overlay", "M"),
                ("Normalize", "show_normalized", "N"),
                ("Image", "show_image", "I"),
            ]

            def _toggle_view(state_key):
                ss[state_key] = not ss.get(state_key, True)

            _view_cols = st.columns(len(_views))
            for _col, (_label, _state_key, _short) in zip(_view_cols, _views):
                ss.setdefault(_state_key, True)
                _col.button(
                    _label,
                    key=f"view_btn_{_state_key}",
                    shortcut=_short,
                    type="primary" if ss[_state_key] else "secondary",
                    on_click=_toggle_view,
                    args=(_state_key,),
                    width="stretch",
                    help=f"Toggle {_label.lower()} (shortcut: {_short})",
                )

            # segmentation / classification controls as tabs: both panels render
            # into the DOM so their button keyboard shortcuts work from either tab
            seg_tab, classify_tab = st.tabs(
                ["Segmentation Controls", "Classification Controls"], width="stretch"
            )
            with seg_tab:
                mask_editing_panel.render_segment_sidebar(key_ns="edit_side")
            with classify_tab:
                mask_editing_panel.render_classify_sidebar(key_ns="classify_side")

            # editing tools available in both tabs
            mask_editing_panel.render_common_tools(key_ns="tools")

    # Zoom/pan panel — sits to the right of the image (col3), but its code runs before
    # the image column so the crop reads the updated zoom on the same run (columns are
    # positioned by creation order, executed in the order the `with` blocks appear).
    with col3:
        mask_editing_panel.render_zoom_panel()

    # Page main content
    with col2:
        mask_editing_panel.render_main(key_ns="edit")
