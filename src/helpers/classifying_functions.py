# panels/classify_cells.py
import numpy as np
import streamlit as st

from src.helpers.state_ops import ordered_keys, get_current_rec
from src.helpers.densenet_functions import (
    classify_cells_with_densenet,
    densenet_mapping_fragment,
)

ss = st.session_state

class_colour_hexes = [
    "#DC050C",
    "#5289C7",
    "#4EB265",
    "#F7F056",
    "#882E72",
    "#E8601C",
    "#D1BBD7",
    "#90C987",
    "#F1932D",
    "#CAE0AB",
    "#F6C141",
]


def hex_to_rgb01(hx: str) -> tuple[float, float, float]:
    """Convert hex color string to (r,g,b) in 0..1 range."""
    return tuple(int(hx[i : i + 2], 16) / 255.0 for i in (1, 3, 5))


def color_hex_for(name: str) -> str:
    """
    Deterministic, unique color per class name using session-backed mapping.
    """
    if name == "No label":
        return "#E5E5E5"
    cmap = st.session_state.setdefault("class_colors", {})
    if name not in cmap:
        # pick the first palette color not currently used; wrap if all are used
        used = set(cmap.values())
        choice = next((hx for hx in class_colour_hexes if hx not in used), None)
        if choice is None:
            choice = class_colour_hexes[len(cmap) % len(class_colour_hexes)]
        cmap[name] = choice
    return cmap[name]


def color_chip_md(hex_color: str, size: int = 14) -> str:
    """Return HTML markdown for a color chip of given hex color and size."""
    return (
        f'<span style="display:inline-block;'
        f"width:{size}px;height:{size}px;margin-top:2px;"
        f"background:{hex_color};border:1px solid rgba(0,0,0,.15);"
        f'border-radius:3px;"></span>'
    )


def rename_class_from_input(old_key: str, new_key: str):
    """Rename class based on input box value; clears input box after use."""
    ss = st.session_state
    old = ss.get(old_key)
    new = (ss.get(new_key, "") or "").strip()
    if not old or not new or old == new:
        return

    # Clear the input box
    ss[new_key] = ""

    # Stash the desired selectbox value for the next run
    ss[f"{old_key}__next_value"] = new

    # Apply the rename (this will st.rerun())
    rename_class_everywhere(old, new)


def yield_all_image_records():
    """Yield all image records from session state safely."""
    ims = st.session_state.get("images", {}) or {}
    for k in ordered_keys():
        rec = ims.get(k)
        if isinstance(rec, dict):
            yield k, rec


def rename_class_everywhere(old_name: str, new_name: str):
    """
    Rename a class globally across the session. If new_name already exists,
    labels with old_name are reassigned to new_name and old_name is removed.
    This updates:
      - st.session_state['all_classes']
      - per-image rec['labels'] dicts
      - st.session_state['side_current_class'] if it was old_name
      - emoji map st.session_state['class_emojis']
    """
    ss = st.session_state
    if not old_name or old_name == "No label":
        st.warning("That class cannot be renamed.")
        return
    new_name = (new_name or "").strip()
    if not new_name or new_name == "No label":
        st.warning("Please choose a non-empty class name that isn't reserved.")
        return
    if new_name == old_name:
        st.info("No change needed.")
        return

    # Ensure class list exists
    all_classes = ss.setdefault("all_classes", ["No label"])

    # Track whether target already exists (merge)
    target_exists = new_name in all_classes

    # Update labels in every image record)
    changed_labels = 0
    for _, rec in yield_all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            # Re-assign values from old_name -> new_name
            to_update = [iid for iid, cname in lab.items() if cname == old_name]
            for iid in to_update:
                lab[iid] = new_name
            changed_labels += len(to_update)

    # Update class list (merge or rename)
    if old_name in all_classes:
        all_classes = [c for c in all_classes if c != old_name]  # drop old
    if new_name not in all_classes:
        all_classes.append(new_name)
    # Keep "No label" first if you prefer; otherwise keep as-is
    # Re-assign back
    ss["all_classes"] = all_classes

    # --- Update current selection if needed ---
    if ss.get("side_current_class") == old_name:
        ss["side_current_class"] = new_name

    # after emoji handling
    cmap = ss.setdefault("class_colors", {})
    if new_name in ss["all_classes"]:  # target_exists logic above already computed
        cmap.pop(old_name, None)  # merge: keep target color
    else:
        if old_name in cmap:
            cmap[new_name] = cmap.pop(old_name)
    st.rerun()


def remove_class_everywhere(name: str):
    """
    Delete a class from the session. All masks with this class are unlabelled (set to None).
    Updates:
      - st.session_state['all_classes'] (removes the class)
      - per-image rec['labels'] values (name -> None)
      - st.session_state['side_current_class'] (fallback to "No label" if needed)
      - emoji map st.session_state['class_emojis'] (removes entry)
    """
    ss = st.session_state

    ss.setdefault("class_emojis", {}).pop(name, None)
    ss.setdefault("class_colors", {}).pop(name, None)  # free color

    # Ensure class list exists and cant remove "No label"
    if name == "No label":
        return
    all_classes = ss.setdefault("all_classes", ["No label"])
    if name not in all_classes:
        return

    # Unlabel everywhere
    changed = 0
    for _, rec in yield_all_image_records():
        lab = rec.get("labels")
        if isinstance(lab, dict):
            to_update = [iid for iid, cname in lab.items() if cname == name]
            for iid in to_update:
                lab[iid] = None
            changed += len(to_update)

    # Remove from class list & emoji map
    ss["all_classes"] = [c for c in all_classes if c != name]
    ss.setdefault("class_emojis", {}).pop(name, None)

    # Fix current selection if needed
    if ss.get("side_current_class") == name:
        ss["side_current_class"] = "No label"
    st.rerun()


def create_colour_palette(class_names):
    """
    Return {class_name: (r,g,b)} in 0..1 using the fixed palette.
    Includes '__unlabeled__' as white.
    """
    pal = {"__unlabeled__": (1.0, 1.0, 1.0)}
    for n in class_names:
        if not n or n == "No label":
            continue
        pal[n] = hex_to_rgb01(color_hex_for(n))
    return pal


def classes_map_from_labels(masks, labels):
    """
    Given masks array and labels dict {inst_id: class_name}, return
    {inst_id: class_name} for all inst_ids in masks (excluding 0
    """
    inst = np.asarray(masks)
    if inst.ndim != 2 or inst.size == 0:
        return {}
    classes_map = {}
    for iid in np.unique(inst):
        if iid == 0:
            continue
        cls = labels.get(int(iid))
        classes_map[int(iid)] = cls if cls not in (None, "") else "No label"
    return classes_map


def create_row(name: str, key: str, mode_ns: str = "side"):
    """
    Create a single class selection row with color chip, name, count, and buttons.
    """
    # icon | name | click-assign | assign-all |
    c1, c2, c3, c4 = st.columns([1, 5, 3, 3])
    c1.markdown(color_chip_md(color_hex_for(name)), unsafe_allow_html=True)
    c2.write(f"**{name}**")

    def _select():
        # pick this class and switch the main panel to Assign class mode
        st.session_state["pending_class"] = name
        st.session_state["interaction_mode"] = "Assign class"

    def _assign_all():
        # assign ALL masks in the current image to this class
        rec = get_current_rec()
        mask_ids = [int(i) for i in np.unique(rec["masks"]) if i != 0]
        rec["labels"] = {mid: name for mid in mask_ids}

    # assigns all masks in this image to this class
    c3.button(
        "All",
        key=f"{key}_assign_all",
        width='stretch',
        on_click=_assign_all,
        help="Set all masks in this image to this class",
    )

    # sets the current assignable class by clicking cells
    c4.button(
        "Click",
        key=f"{key}_select",
        width='stretch',
        on_click=_select,
        help="Click masks to label cells",
    )


def add_label_from_input(labels, new_label_ss):
    """Add a new label from input box; clears input box after use."""
    new_label = new_label_ss.strip()
    if not new_label:
        return
    # assumes `labels` is in scope; consider storing it in st.session_state if needed
    if new_label not in labels:
        labels.append(new_label)
    st.session_state["side_current_class"] = new_label
    st.session_state["side_new_label"] = ""
    st.rerun()


# -----------------------------------------------------#
# ------------- CLASSIFY SIDEBAR ACTIONS -------------- #
# -----------------------------------------------------#


def densenet_help(has_model, needs_mapping):
    if not has_model:
        if needs_mapping:
            return "All predictions are mapped to 'No label'. Add some classes under 'Manage Classes' and map them to the predictions below."
        return "Classify all masks in this image with the loaded Densenet121 model."
    return "Upload or fine-tune a Densenet model before auto-classifying cells."


@st.fragment
def classify_actions_fragment():

    needs_mapping = all(
        label == "No label" for label in st.session_state["densenet_class_map"].values()
    )

    rec = get_current_rec()

    # buttons to classify masks in current image or batch classify all images
    col1, col2 = st.columns(2)
    with col1:
        help = densenet_help(st.session_state["densenet_model"] == None, needs_mapping)
        # classify masks in the current image
        if st.button(
            "Classify",
            width='stretch',
            help=help,
            disabled=(st.session_state["densenet_model"] == None) or needs_mapping,
        ):
            classify_cells_with_densenet(rec)
            st.rerun()
    with col2:
        # batch classify masks in all images
        if st.button(
            "Batch classify",
            key="btn_batch_classify_cellpose",
            width='stretch',
            help=help,
            disabled=st.session_state["densenet_model"] == None or needs_mapping,
        ):
            batch_classify()
            st.rerun()

    # mapping fragment for assiging model outputs to classes
    with st.expander("Map model predictions to cell classes"):
        densenet_mapping_fragment()


def batch_classify():
    """classify masks in the all images"""
    ok = ordered_keys()
    n = len(ok)
    pb = st.progress(0.0, text="Starting…")
    for i, k in enumerate(ok, 1):
        classify_cells_with_densenet(st.session_state.images.get(k))
        pb.progress(i / n, text=f"Classified {i}/{n}")


# -----------------------------------------------------#
# ------------- RENDER CLASSIFY SIDEBAR -------------- #
# -----------------------------------------------------#


def class_selection_fragment():

    # Create class selection rows
    ss = st.session_state
    if "pending_class" in ss:
        pc = ss.pop("pending_class")
        if pc not in ss["all_classes"]:
            ss["all_classes"].append(pc)
        ss["side_current_class"] = pc
    ss.setdefault("side_current_class", ss["all_classes"][0])

    rec = get_current_rec()
    labels = ss.setdefault("all_classes", ["No label"])

    # Unlabeled row
    create_row("No label", key="use_unlabel")

    # Actual classes
    for name in [c for c in labels if c != "No label"]:
        create_row(name, key=f"use_{name}")


def class_manage_fragment(key_ns="side"):
    ss = st.session_state
    labels = ss.setdefault("all_classes", ["No label"])

    # Apply any pending rename select value before creating widgets
    pending_key = f"{key_ns}_rename_from__next_value"
    pending_val = ss.pop(pending_key, None)
    if pending_val is not None:
        ss[f"{key_ns}_rename_from"] = pending_val

    # add new class by typing in the text box
    st.text_input(
        "Add a new class:",
        key="side_new_label",
        placeholder="New class name",
        on_change=add_label_from_input(labels, ss.get("side_new_label", "")),
    )

    # delete class by selecting from dropdown list
    editable = [c for c in labels if c != "No label"]
    if editable:
        del_class = st.selectbox(
            "Delete a class:",
            options=["Select a class"] + editable,
            key=f"{key_ns}_delete_label",
        )
        if del_class != "Select a class":
            remove_class_everywhere(del_class)
            st.success(f"Deleted class: {del_class}")
            st.rerun()
    else:
        st.caption("No classes yet. Add a class above first.")
        return

    # rename class by selecting from dropdown and typing new name
    c1, c2 = st.columns([1, 2])
    with c1:
        st.selectbox("Rename:", options=editable, key=f"{key_ns}_rename_from")
    with c2:
        st.text_input(
            "To:",
            key=f"{key_ns}_rename_to",
            placeholder="New name",
            on_change=lambda: rename_class_from_input(
                f"{key_ns}_rename_from", f"{key_ns}_rename_to"
            ),
        )
