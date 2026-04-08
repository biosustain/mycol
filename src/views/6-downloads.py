import io
import os
import tempfile
import streamlit as st
import numpy as np
import pandas as pd
import tifffile as tiff
from pathlib import Path
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
has_cellpose_model = bool(ss.get("cellpose_model_bytes"))
has_densenet_model = ss.get("densenet_model") is not None


def _invalidate():
    ss.pop("_dl_bytes", None)


def _invalidate_session():
    ss.pop("_session_bytes", None)


st.header("Download annotated images, cell metrics, and trained models")

if not has_images and not has_cellpose and not has_densenet:
    st.info("Upload data and train models first.")
    st.stop()


# ── Options ───────────────────────────────────────────────────────────────────
def _row(label, default, key, caption, disabled=False, on_change=_invalidate):
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    with c1:
        st.checkbox(label, default, key=key, on_change=on_change, disabled=disabled)
    with c2:
        st.caption(caption)


with st.container(border=True):
    img_col, table_col, model_col, session_col = st.columns(4)

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

    with session_col:
        st.subheader("Session Restore")
        _row(
            "Save Session",
            has_images,
            "dl_save_session",
            "Reupload this zip to restore your session.",
            disabled=not has_images,
            on_change=_invalidate_session,
        )


# ── Build main zip ────────────────────────────────────────────────────────────
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


# ── Build session zip ─────────────────────────────────────────────────────────
def _build_session_zip() -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:

        # Images and masks
        for k in ok:
            rec = images[k]
            name = Path(rec.get("name", str(k))).stem

            ibuf = io.BytesIO()
            tiff.imwrite(
                ibuf, np.asarray(rec["image"]), photometric="rgb", compression="deflate"
            )
            zf.writestr(f"images/{name}.tif", ibuf.getvalue())

            mask = rec.get("masks")
            if mask is not None:
                mbuf = io.BytesIO()
                tiff.imwrite(mbuf, mask.astype(np.uint16))
                zf.writestr(f"masks/{name}.tif", mbuf.getvalue())

        # Cellpose training hyperparameters
        cp_training_params = {
            "base_model": ss.get("cp_base_model"),
            "max_epoch": ss.get("cp_max_epoch"),
            "learning_rate": ss.get("cp_learning_rate"),
            "weight_decay": ss.get("cp_weight_decay"),
            "batch_size": ss.get("cp_batch_size"),
            "min_cells_per_image": ss.get("cp_min_cells_per_image"),
        }
        zf.writestr(
            "cellpose_training_hyperparameters.csv",
            pd.Series(cp_training_params)
            .rename_axis("parameter")
            .reset_index(name="value")
            .to_csv(index=False),
        )

        # Cellpose inference hyperparameters
        cp_inference_params = {
            "channel_1": ss.get("cp_ch1"),
            "channel_2": ss.get("cp_ch2"),
            "diameter": ss.get("cp_diameter"),
            "cellprob_threshold": ss.get("cp_cellprob_threshold"),
            "flow_threshold": ss.get("cp_flow_threshold"),
            "min_size": ss.get("cp_min_size"),
            "niter": ss.get("cp_niter"),
        }
        zf.writestr(
            "cellpose_inference_hyperparameters.csv",
            pd.Series(cp_inference_params)
            .rename_axis("parameter")
            .reset_index(name="value")
            .to_csv(index=False),
        )

        # Fine-tuned Cellpose model
        if has_cellpose_model:
            zf.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])

        # Fine-tuned DenseNet model
        if has_densenet_model:
            import torch

            tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            tmp_path = tmp.name
            tmp.close()
            torch.save(ss["densenet_model"].state_dict(), tmp_path)
            with open(tmp_path, "rb") as f:
                zf.writestr("densenet_model.pth", f.read())
            os.remove(tmp_path)

        # Cell metrics
        zf.writestr(
            "cell_metrics.csv",
            build_cell_metrics_csv(tuple(ss.get("analysis_labels") or ())),
        )

    return buf.getvalue()


# ── Download buttons ──────────────────────────────────────────────────────────
save_session = ss.get("dl_save_session", has_images)

btn_col, dl_col, session_dl_col = st.columns(3)

if btn_col.button("Prepare Download", type="primary", width="stretch"):
    with st.spinner("Preparing download..."):
        ss["_dl_bytes"] = _build_zip()
        if save_session:
            ss["_session_bytes"] = _build_session_zip()

dl_col.download_button(
    "Download Files",
    data=ss.get("_dl_bytes", b""),
    file_name="mycol_downloads.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
    disabled="_dl_bytes" not in ss,
)

session_dl_col.download_button(
    "Download Session Restore",
    data=ss.get("_session_bytes", b""),
    file_name="saved_session.zip",
    mime="application/zip",
    width="stretch",
    type="primary",
    disabled="_session_bytes" not in ss,
)
