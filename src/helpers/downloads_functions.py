import io
import json
import os
import tempfile
import numpy as np
import pandas as pd
import tifffile as tiff
import streamlit as st
import plotly.io as pio
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED, ZIP_STORED

from src.helpers.upload_download_functions import build_masks_images_zip
from src.helpers.cell_metrics_functions import build_cell_metrics_csv
from src.helpers.cellpose_functions import build_cellpose_zip_bytes
from src.helpers.densenet_functions import build_densenet_zip_bytes

ss = st.session_state


SESSION_ZIP_NAME = "mycol_saved_session.zip"


def build_download_zip(images, ok) -> tuple[bytes, str]:
    """Build the download and return its bytes plus the file name to serve it as."""
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:

        if images and ss.get("dl_include_images", True):
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
        elif images and ss.get("dl_include_summary", True):
            inner = build_masks_images_zip(images, ok, False, False, False, True)
            with ZipFile(io.BytesIO(inner)) as inner_zf:
                zf.writestr("cell_counts_per_image.csv", inner_zf.read("cell_counts_per_image.csv"))

        if images and ss.get("dl_include_cell_metrics", True):
            zf.writestr(
                "cell_metrics.csv",
                build_cell_metrics_csv(tuple(ss.get("analysis_labels") or ())),
            )

        # nested zips are already compressed, so they are stored rather than deflated
        if ss.get("dl_include_cellpose", True) and (ss.get("cp_zip_bytes") or ss.get("cellpose_model_bytes")):
            cp_bytes = ss.get("cp_zip_bytes") or build_cellpose_zip_bytes()
            if cp_bytes:
                zf.writestr("cellpose_training.zip", cp_bytes, compress_type=ZIP_STORED)

        if ss.get("dl_include_densenet", True) and (ss.get("dn_zip_bytes") or ss.get("densenet_model") is not None):
            dn_bytes = ss.get("dn_zip_bytes") or build_densenet_zip_bytes(ss.get("dn_input_size"))
            if dn_bytes:
                zf.writestr("densenet_training.zip", dn_bytes, compress_type=ZIP_STORED)

        if images and ss.get("dl_save_session", True):
            zf.writestr(SESSION_ZIP_NAME, build_session_zip(images, ok), compress_type=ZIP_STORED)

    data = buf.getvalue()

    # a session-only download is served unwrapped
    with ZipFile(io.BytesIO(data)) as zf:
        if zf.namelist() == [SESSION_ZIP_NAME]:
            return zf.read(SESSION_ZIP_NAME), SESSION_ZIP_NAME

    return data, "mycol_downloads.zip"


def build_session_zip(images, ok) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:

        mask_suffix = ss.get("mask_suffix", "_masks")
        image_metadata = {}
        for k in ok:
            rec = images[k]
            name = Path(rec.get("name", str(k))).stem

            ibuf = io.BytesIO()
            tiff.imwrite(ibuf, np.asarray(rec["image"]), photometric="rgb", compression="deflate")
            zf.writestr(f"images/{name}.tif", ibuf.getvalue())

            mask = rec.get("masks")
            if mask is not None:
                mbuf = io.BytesIO()
                tiff.imwrite(mbuf, mask.astype(np.uint16))
                zf.writestr(f"masks/{name}{mask_suffix}.tif", mbuf.getvalue())

            image_metadata[name] = {
                "orig_H": rec.get("orig_H", rec["H"]),
                "orig_W": rec.get("orig_W", rec["W"]),
                "boxes": rec.get("boxes", []),
                "boxes_display": rec.get("boxes_display", []),
            }

        zf.writestr(
            "image_metadata.json",
            json.dumps({"mask_suffix": mask_suffix, "images": image_metadata}),
        )

        cp_training_params = {
            "base_model": ss.get("cp_base_model"),
            "max_epoch": ss.get("cp_max_epoch"),
            "learning_rate": ss.get("cp_learning_rate"),
            "weight_decay": ss.get("cp_weight_decay"),
            "batch_size": ss.get("cp_batch_size"),
            "min_cells_per_image": ss.get("cp_min_cells_per_image"),
            # images are converted to grayscale before Cellpose, so channels are fixed
            "training_ch1": 0,
            "training_ch2": 0,
            "do_gridsearch": ss.get("cp_do_gridsearch", False),
            "n_trials": ss.get("cp_n_trials", 20),
        }
        zf.writestr(
            "cellpose_training_hyperparameters.csv",
            pd.Series(cp_training_params).rename_axis("parameter").reset_index(name="value").to_csv(index=False),
        )

        dn_training_params = {
            "input_size": ss.get("dn_input_size", 64),
            "batch_size": ss.get("dn_batch_size", 32),
            "max_epoch": ss.get("dn_max_epoch", 100),
            "val_split": ss.get("dn_val_split", 0.2),
        }
        zf.writestr(
            "densenet_training_hyperparameters.csv",
            pd.Series(dn_training_params).rename_axis("parameter").reset_index(name="value").to_csv(index=False),
        )

        cp_inference_params = {
            # images are converted to grayscale before Cellpose, so channels are fixed
            "channel_1": 0,
            "channel_2": 0,
            "diameter": ss.get("cp_diameter"),
            "cellprob_threshold": ss.get("cp_cellprob_threshold"),
            "flow_threshold": ss.get("cp_flow_threshold"),
            "min_size": ss.get("cp_min_size"),
            "niter": ss.get("cp_niter"),
        }
        zf.writestr(
            "cellpose_inference_hyperparameters.csv",
            pd.Series(cp_inference_params).rename_axis("parameter").reset_index(name="value").to_csv(index=False),
        )

        if bool(ss.get("cellpose_model_bytes")):
            zf.writestr("cellpose_model.pt", ss["cellpose_model_bytes"])

        if ss.get("densenet_model") is not None:
            import torch
            tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            tmp_path = tmp.name
            tmp.close()
            torch.save(ss["densenet_model"].state_dict(), tmp_path)
            with open(tmp_path, "rb") as f:
                zf.writestr("densenet_model.pth", f.read())
            os.remove(tmp_path)

        densenet_class_map = ss.get("densenet_class_map")
        if densenet_class_map:
            zf.writestr(
                "densenet_class_map.csv",
                pd.DataFrame(
                    [{"class_index": k, "class_name": v} for k, v in densenet_class_map.items()]
                ).to_csv(index=False),
            )

        zf.writestr(
            "cell_metrics.csv",
            # session snapshot: persist ALL masks/labels, ignoring the Cell
            # Metrics comparison filter (otherwise excluded labels are lost on restore)
            build_cell_metrics_csv(()),
        )

        for fig_key, filename in [
            ("cellpose_training_losses", "cellpose_training_losses.json"),
            ("cellpose_iou_comparison", "cellpose_iou_comparison.json"),
            ("cellpose_original_counts_comparison", "cellpose_original_counts_comparison.json"),
            ("cellpose_tuned_counts_comparison", "cellpose_tuned_counts_comparison.json"),
            ("densenet_training_losses", "densenet_training_losses.json"),
            ("densenet_training_metrics", "densenet_training_metrics.json"),
            ("densenet_confusion_matrix", "densenet_confusion_matrix.json"),
        ]:
            fig = ss.get(fig_key)
            if fig is not None:
                zf.writestr(filename, pio.to_json(fig))

        cp_grid_results_df = ss.get("cp_grid_results_df")
        if cp_grid_results_df is not None:
            zf.writestr("cellpose_grid_search_results.csv", cp_grid_results_df.to_csv(index=False))

    return buf.getvalue()
