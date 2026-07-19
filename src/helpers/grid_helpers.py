# helpers/grid_helpers.py
"""Shared image-selection table used by the training, Cell Metrics and uploads pages."""
import numpy as np
import pandas as pd
import streamlit as st

from src.helpers.state_ops import ordered_keys

ss = st.session_state


def _n_masks(rec) -> int:
    """Number of distinct cell masks in an image record."""
    m = rec.get("masks")
    if isinstance(m, np.ndarray) and m.ndim == 2 and np.any(m):
        return int(len(np.unique(m)) - 1)
    return 0


def _n_labelled(rec) -> int:
    """Number of masks that carry a real class label."""
    labels = rec.get("labels") or {}
    return sum(1 for v in labels.values() if v not in (None, "No label"))


DISPLAY_COLS = ["No.", "Image", "Mask Present", "Number of Masks", "Labelled Masks"]


def _build_image_table(keys) -> pd.DataFrame:
    """Per-image summary rows in the same format as the uploads-page table."""
    rows = []
    for i, k in enumerate(keys, start=1):
        rec = ss["images"][k]
        n = _n_masks(rec)
        nl = _n_labelled(rec)
        rows.append(
            {
                "_key": k,
                "No.": i,
                "Image": rec.get("name", k),
                "Mask Present": "✅" if n > 0 else "❌",
                "Number of Masks": n,
                "Labelled Masks": f"{nl}/{n}",
            }
        )
    return pd.DataFrame(rows)


def render_image_selection_table(namespace: str, default_all: bool = True) -> list:
    """Multi-select image table showing mask and labelled-mask counts.

    Shared by the training, Cell Metrics and uploads pages so they look identical.
    Persists the user's choice in ``ss[f'{namespace}_selected_image_keys']`` and
    returns the ordered list of selected image keys. Untouched images follow
    ``default_all``.
    """
    keys = ordered_keys()
    if not keys:
        ss[f"{namespace}_selected_image_keys"] = []
        return []

    df = _build_image_table(keys)
    ids = df["_key"].tolist()

    # Ticks stored per image key; survives page navigation.
    sel_key = f"{namespace}_selection"
    sel = ss.setdefault(sel_key, {})

    # Selections are row positions and never auto-reset, so a removal would strand
    # ticks on the wrong images: rebuild the widget whenever the row set changes.
    grid_key = f"{namespace}_image_grid_{hash(tuple(ids))}"

    event = st.dataframe(
        df[DISPLAY_COLS],
        hide_index=True,
        width="stretch",
        height=min(400, 38 + 35 * len(ids)),
        on_select="rerun",
        selection_mode="multi-row",
        # Seeds the ticks when the widget has no state of its own.
        selection_default={
            "selection": {
                "rows": [i for i, k in enumerate(ids) if sel.get(k, default_all)]
            }
        },
        key=grid_key,
    )

    chosen = set(event.selection.rows)
    ss[sel_key] = {k: i in chosen for i, k in enumerate(ids)}

    selected = [k for i, k in enumerate(ids) if i in chosen]
    ss[f"{namespace}_selected_image_keys"] = selected
    return selected
