import streamlit as st
from src.helpers.state_ops import ordered_keys, require_images
from src.helpers.cell_metrics_functions import METRIC_COLS
from src.helpers.downloads_functions import build_download_zip
from src.panels.downloads_panel import option_row

ss = st.session_state
images = ss.get("images", {})
ok = ordered_keys()
has_images = bool(ok and images)
has_cellpose = "cp_zip_bytes" in ss or bool(ss.get("cellpose_model_bytes"))
has_densenet = "dn_zip_bytes" in ss or ss.get("densenet_model") is not None

require_images()

st.header("Download annotated images, cell metrics, and trained models")


# ── Options ────────────────────────────────────────────────────────────────────
def _invalidate():
    ss.pop("_dl_bytes", None)
    ss.pop("_dl_name", None)


img_col, table_col, model_col, session_col = st.columns(4, border=True)

with img_col:
    st.subheader("Images and Masks", text_alignment="center")
    if has_images:
        include_images = option_row("Images and masks", True, "dl_include_images", "The image files and their segmentation mask files.", on_change=_invalidate)
        option_row("Colored mask overlays", True, "dl_include_overlay", "Color-coded regions drawn over each image to visualize the segmented cells.", disabled=not include_images, on_change=_invalidate)
        option_row("Per-image class counts", False, "dl_include_counts", "Print the number of cells per class in the corner of each image.", disabled=not include_images, on_change=_invalidate)
        option_row("Normalize images", False, "dl_normalize_download", "Rescale pixel intensities to span the full 0–255 range before saving.", disabled=not include_images, on_change=_invalidate)
        option_row("Cell patch images", False, "dl_include_patches", "Save a cropped image for every individual segmented cell.", disabled=not include_images, on_change=_invalidate)
    else:
        st.caption("No images uploaded yet.")

with table_col:
    st.subheader("Tables", text_alignment="center")
    if has_images:
        option_row("Per-image cell counts", True, "dl_include_summary", "CSV listing how many cells of each class appear in each image.", on_change=_invalidate)
        option_row("Cell metrics", True, "dl_include_cell_metrics", f"CSV of {len(METRIC_COLS)} morphological descriptors ({', '.join(METRIC_COLS[:3])}, etc.) for every cell.", on_change=_invalidate)
    else:
        st.caption("No data available yet.")

with model_col:
    st.subheader("Trained Models", text_alignment="center")
    option_row("Cellpose model", has_cellpose, "dl_include_cellpose", "Fine-tuned model weights, the training dataset, and loss curves.", disabled=not has_cellpose, on_change=_invalidate)
    option_row("DenseNet model", has_densenet, "dl_include_densenet", "Fine-tuned model weights, the training dataset, and evaluation metrics.", disabled=not has_densenet, on_change=_invalidate)

with session_col:
    st.subheader("Session Restore", text_alignment="center")
    option_row("Save Session", has_images, "dl_save_session", "Adds mycol_saved_session.zip to the download; reupload that zip to restore your session.", disabled=not has_images, on_change=_invalidate)


# ── Download button ────────────────────────────────────────────────────────────
btn_col, dl_col = st.columns(2)

if btn_col.button("Prepare Download", type="primary", width="stretch"):
    with st.spinner("Preparing download..."):
        ss["_dl_bytes"], ss["_dl_name"] = build_download_zip(images, ok)

dl_col.download_button(
    "Download Files",
    data=ss.get("_dl_bytes", b""),
    file_name=ss.get("_dl_name", "mycol_downloads.zip"),
    mime="application/zip",
    width="stretch",
    type="primary",
    disabled="_dl_bytes" not in ss,
)
