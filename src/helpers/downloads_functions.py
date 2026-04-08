import io
import json
import os
import tempfile
import numpy as np
import pandas as pd
import tifffile as tiff
import streamlit as st
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from src.helpers.upload_download_functions import build_masks_images_zip
from src.helpers.cell_metrics_functions import build_cell_metrics_csv

ss = st.session_state


def build_download_zip(images, ok) -> bytes:
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

            if ss.get("dl_include_cell_metrics", True):
                zf.writestr(
                    "cell_metrics.csv",
                    build_cell_metrics_csv(tuple(ss.get("analysis_labels") or ())),
                )

        if "cp_zip_bytes" in ss and ss.get("dl_include_cellpose", True):
            zf.writestr("cellpose_training.zip", ss["cp_zip_bytes"])

        if "dn_zip_bytes" in ss and ss.get("dl_include_densenet", True):
            zf.writestr("densenet_training.zip", ss["dn_zip_bytes"])

    return buf.getvalue()


def build_session_zip(images, ok) -> bytes:
    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:

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
                zf.writestr(f"masks/{name}.tif", mbuf.getvalue())

            image_metadata[name] = {
                "orig_H": rec.get("orig_H", rec["H"]),
                "orig_W": rec.get("orig_W", rec["W"]),
                "boxes": rec.get("boxes", []),
                "boxes_display": rec.get("boxes_display", []),
            }

        zf.writestr("image_metadata.json", json.dumps(image_metadata))

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
            pd.Series(cp_training_params).rename_axis("parameter").reset_index(name="value").to_csv(index=False),
        )

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
            build_cell_metrics_csv(tuple(ss.get("analysis_labels") or ())),
        )

    return buf.getvalue()
