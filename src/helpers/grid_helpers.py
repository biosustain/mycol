# helpers/grid_helpers.py
"""Shared image-selection table used by the training, Cell Metrics and uploads pages."""
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from src.helpers.state_ops import ordered_keys


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
        rec = st.session_state["images"][k]
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


def _selected_keys_from_grid(grid_response) -> set:
    """Extract the set of selected ``_key`` values from an AgGrid response."""
    selected_rows = grid_response.get("selected_rows")
    if selected_rows is None:
        return set()
    if isinstance(selected_rows, pd.DataFrame):
        return set(selected_rows.get("_key", pd.Series([], dtype=str)).tolist())
    if isinstance(selected_rows, list):
        if selected_rows and isinstance(selected_rows[0], dict):
            return {row.get("_key") for row in selected_rows if row.get("_key")}
        return {row for row in selected_rows if isinstance(row, str)}
    return set()


def render_image_selection_table(namespace: str) -> list:
    """Multi-select image table showing mask and labelled-mask counts.

    Shared by the training, Cell Metrics and uploads pages so they look identical.
    Persists the user's choice in ``ss[f'{namespace}_selected_image_keys']`` and
    returns the ordered list of selected image keys.
    """
    keys = ordered_keys()
    if not keys:
        st.session_state[f"{namespace}_selected_image_keys"] = []
        return []

    df = _build_image_table(keys)
    ids = df["_key"].tolist()

    display_df = df[DISPLAY_COLS + ["_key"]].copy()
    gb = GridOptionsBuilder.from_dataframe(display_df)
    # single checkbox column: the select-all header + row checkboxes go on the
    # first column ("No."); no other column gets its own checkbox
    gb.configure_selection(
        "multiple", use_checkbox=True, header_checkbox=True, rowMultiSelectWithClick=True
    )
    gb.configure_column("_key", hide=True)
    for col in DISPLAY_COLS:
        gb.configure_column(col, editable=False, type=[])

    grid_response = AgGrid(
        display_df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=min(400, 55 + 35 * len(ids)),
        width="100%",
        key=f"{namespace}_image_grid",
    )

    selected_keys = _selected_keys_from_grid(grid_response)
    selected = [k for k in ids if k in selected_keys]
    st.session_state[f"{namespace}_selected_image_keys"] = selected
    return selected
