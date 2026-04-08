import io
import streamlit as st
from zipfile import ZipFile, ZIP_DEFLATED

from src.helpers.upload_download_functions import build_masks_images_zip
from src.helpers.cell_metrics_functions import build_cell_metrics_csv
from src.helpers.state_ops import ordered_keys

ss = st.session_state
images = ss.get("images", {})
ok = ordered_keys()
has_images = bool(ok and images)
has_cellpose = "cp_zip_bytes" in ss
has_densenet = "dn_zip_bytes" in ss


def _invalidate():
    ss.pop("_dl_bytes", None)


st.header("Download annotated images, cell metrics, and trained models")

if not has_images and not has_cellpose and not has_densenet:
    st.info("Upload data and train models first.")
    st.stop()


# ── Options ──────────────────────────────────────────────────────────────────
def _row(label, default, key, caption, disabled=False):
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        st.checkbox(label, default, key=key, on_change=_invalidate, disabled=disabled)
    with c2:
        st.caption(caption)


with st.container(border=True):
    img_col, table_col, model_col = st.columns(3)

    with img_col:
        h_col, cb_col, _ = st.columns([2, 1, 1], vertical_alignment="center")
        h_col.subheader("Images and Masks")
        include_images = cb_col.checkbox(
            "Include",
            has_images,
            key="dl_include_images",
            on_change=_invalidate,
            disabled=not has_images,
        )
        if has_images:
            _row(
                "Colored mask overlays",
                True,
                "dl_include_overlay",
                "Color-coded regions drawn over each image to visualize the segmented cells.",
                disabled=not include_images,
            )
            _row(
                "Per-image class counts",
                False,
                "dl_include_counts",
                "Print the number of cells per class in the corner of each image.",
                disabled=not include_images,
            )
            _row(
                "Normalize images",
                False,
                "dl_normalize_download",
                "Rescale pixel intensities to span the full 0–255 range before saving.",
                disabled=not include_images,
            )
            _row(
                "Cell patch images",
                False,
                "dl_include_patches",
                "Save a cropped image for every individual segmented cell.",
                disabled=not include_images,
            )
        else:
            st.caption("No images uploaded yet.")

    with table_col:
        st.subheader("Tables")
        if has_images:
            _row(
                "Per-image cell counts",
                True,
                "dl_include_summary",
                "CSV listing how many cells of each class appear in each image.",
            )
            _row(
                "Cell metrics",
                True,
                "dl_include_cell_metrics",
                "CSV of morphological descriptors (area, circularity, elongation, etc.) for every cell.",
            )
        else:
            st.caption("No data available yet.")

    with model_col:
        st.subheader("Trained Models")
        _row(
            "Cellpose model",
            has_cellpose,
            "dl_include_cellpose",
            "Fine-tuned model weights, the training dataset, and loss curves.",
            disabled=not has_cellpose,
        )
        _row(
            "DenseNet model",
            has_densenet,
            "dl_include_densenet",
            "Fine-tuned model weights, the training dataset, and evaluation metrics.",
            disabled=not has_densenet,
        )


# ── Build zip ─────────────────────────────────────────────────────────────────
def _build_zip() -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:

        if has_images and ss.get("dl_include_images", True):
            inner = build_masks_images_zip(
                images,
                ok,
                ss.get("dl_include_overlay", True),
                ss.get("dl_include_counts", False),
                ss.get("dl_include_patches", False),
                ss.get("dl_include_summary", True),
            )
            with ZipFile(io.BytesIO(inner)) as inner_zf:
                for name in inner_zf.namelist():
                    zf.writestr(name, inner_zf.read(name))

            if ss.get("dl_include_cell_metrics", True):
                zf.writestr(
                    "cell_metrics.csv",
                    build_cell_metrics_csv(tuple(ss.get("analysis_labels") or ())),
                )

        if has_cellpose and ss.get("dl_include_cellpose", True):
            zf.writestr("cellpose_training.zip", ss["cp_zip_bytes"])

        if has_densenet and ss.get("dl_include_densenet", True):
            zf.writestr("densenet_training.zip", ss["dn_zip_bytes"])

    return buf.getvalue()


# ── Download buttons ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

if col1.button("Prepare Download", type="primary", width="stretch"):
    with st.spinner("Preparing download..."):
        ss["_dl_bytes"] = _build_zip()

col2.download_button(
    "Download Files",
    data=ss.get("_dl_bytes", b""),
    file_name="mycol_downloads.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
    disabled="_dl_bytes" not in ss,
)
