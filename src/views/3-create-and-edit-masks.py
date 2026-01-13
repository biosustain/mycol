import streamlit as st
from src.panels import mask_editing_panel
from src.helpers.state_ops import ordered_keys, set_current_by_index


# Warning if no images have been uploaded yet
if st.session_state["images"] == {}:
    st.warning("⚠️ Please upload an image on the 'Upload Models and Data' tab first.")
    st.stop()

with st.spinner("Loading Annotator..."):
    col1, col2 = st.columns([2, 5])
    with col1:

        with st.container(border=True, height=770):

            ok = ordered_keys()
            names = [st.session_state.images[k]["name"] for k in ok]

            # --- Ensure we have a current key ---
            if (
                "current_key" not in st.session_state
                or st.session_state.current_key not in ok
            ):
                st.session_state.current_key = ok[0]

            reck = st.session_state.current_key
            rec_idx = ok.index(reck) if reck in ok else 0  # 0-based index

            st.info(f"**Image {rec_idx+1}/{len(ok)}:** {names[rec_idx]}")

            # removed slider and buttons if only one image to prevent crash
            if len(ok) != 1:

                # --- Initialize slider state from current image (first run only) ---
                if "slider_jump" not in st.session_state:
                    # slider is 1-based, rec_idx is 0-based
                    st.session_state.slider_jump = rec_idx + 1

                # --- Helper: keep current_key in sync with slider value ---
                def set_current_from_slider():
                    ok_local = ordered_keys()
                    # slider is 1..len(ok), convert to 0-based index, clamp to range
                    idx = max(
                        0, min(len(ok_local) - 1, st.session_state.slider_jump - 1)
                    )
                    set_current_by_index(idx)

                # --- Navigation buttons & slider ---
                nav_col1, nav_col2, nav_col3 = st.columns([1, 4, 1])

                with nav_col1:
                    if st.button("", width="stretch", shortcut="left"):
                        # move slider one step back, then update current image
                        st.session_state.slider_jump = max(
                            1, st.session_state.slider_jump - 1
                        )
                        set_current_from_slider()
                        st.rerun()

                with nav_col3:
                    if st.button("", width="stretch", shortcut="right"):
                        # move slider one step forward, then update current image
                        st.session_state.slider_jump = min(
                            len(ok), st.session_state.slider_jump + 1
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

            # --- Toggles for overlay and normalization ---
            inner_col1, inner_col2, inner_col3 = st.columns([4, 5, 4])
            with inner_col1:
                # toggle to show/hide masks overlay
                show_overlay_toggle = st.toggle(
                    "Masks",
                    key="show_overlay_w",
                    value=st.session_state.get("show_overlay", True),
                )
                st.session_state["show_overlay"] = show_overlay_toggle

            with inner_col2:
                # toggle to normalize background image
                normalize_image_toggle = st.toggle(
                    "Normalize",
                    key="show_normalized_w",
                    value=st.session_state.get("show_normalized", True),
                )
                st.session_state["show_normalized"] = normalize_image_toggle

            with inner_col3:
                show_image_toggle = st.toggle(
                    "Image",
                    key="show_image_w",
                    value=st.session_state.get("show_image", True),
                )
                st.session_state["show_image"] = show_image_toggle

            # Tabs for editing and classifying masks
            listTabs = ["Segment Cells", "Classify Cells"]
            whitespace = 9
            editing_tab, classifying_tab = st.tabs(["Segment Cells", "Classify Cells"])
            with editing_tab:
                mask_editing_panel.render_segment_sidebar(key_ns="edit_side")
            with classifying_tab:
                mask_editing_panel.render_classify_sidebar(key_ns="classify_side")

            mask_editing_panel.render_download_button()

    # Page main content
    with col2:
        mask_editing_panel.render_main(key_ns="edit")
