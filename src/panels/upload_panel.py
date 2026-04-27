import streamlit as st
from src.helpers.state_ops import (
    ordered_keys,
)
from src.helpers.upload_download_functions import (
    process_uploads,
    render_images_form,
    load_demo_data,
    restore_session,
)
import os
import tempfile
import hashlib
import pandas as pd


def render_main():

    ss = st.session_state

    # briefly show any skipped files that threw an error
    skipped = ss.pop("skipped_files", None)
    if skipped:
        st.toast(
            "**The following files could not be uploaded:**  \n"
            + "  \n".join(f"• {f}" for f in skipped),
            duration="infinite",
        )

    IMAGE_EXTS = {".tif", ".tiff", ".npy", ".png", ".jpg", ".jpeg"}
    MODEL_EXTS = {".pt", ".pth"}

    with st.container(border=True):
        st.subheader("Upload Images, Masks and Models")
        st.caption(
            "Upload images (.tif, .png, .jpg), to begin analysis. Optionally, masks,"
            "segmentation models and classification models can also be uploaded or a previous "
            "session zip file can be uploaded to restore your session."
            "Masks must share the image filename plus a suffix (default: _masks)."
        )

        up_key = f"u_all_{ss.get('uploader_nonce', 0)}"
        files = st.file_uploader(
            " ",
            type=[
                "tif",
                "tiff",
                "npy",
                "png",
                "jpg",
                "jpeg",
                "pt",
                "pth",
                "csv",
                "zip",
            ],
            accept_multiple_files=True,
            key=up_key,
            help="Images and masks are paired by filename. "
            "Model type is detected from file contents. "
            "Unrecognised masks or masks without a paired image will be ignored.",
        )

        # options row
        suffix_col, resize_col, demo_col = st.columns(
            [2, 1, 1], vertical_alignment="bottom"
        )
        with suffix_col:
            mask_suffix = st.text_input(
                "Mask suffix",
                value=ss.get("mask_suffix", "_masks"),
                key="mask_suffix_input",
                help="Mask file names must match uploaded image name plus this suffix",
            )
            ss["mask_suffix"] = mask_suffix.strip() or "_masks"
        with resize_col:
            resize_on_upload = st.checkbox(
                "Resize (512x512)",
                value=ss.get("resize_on_upload", True),
                key="resize_on_upload_checkbox",
                help="Resize images & masks while uploading to 512x512. Recommended for large images, "
                "but may cause quality loss. Calculated cell properties will still be correct for the "
                "original (pre-resize) image.",
            )
            ss["resize_on_upload"] = resize_on_upload
        with demo_col:
            if st.button("Use demo data", type="primary", width="stretch"):
                load_demo_data()

        if files:
            max_bytes = 512 * 1024 * 1024
            oversized = [f for f in files if f.size > max_bytes]
            if oversized:
                total_mb = sum(f.size for f in files) / (1024 * 1024)
                st.toast(
                    f"Error: Upload too large ({total_mb:.0f} MB). Limit is 512 MB. Upload in batches.",
                )
                ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
                st.rerun()

            import torch
            from src.helpers.densenet_functions import build_densenet

            zip_files = [f for f in files if f.name.lower().endswith(".zip")]
            image_files = [
                f for f in files if os.path.splitext(f.name)[1].lower() in IMAGE_EXTS
            ]
            model_files = [
                f
                for f in files
                if os.path.splitext(f.name)[1].lower() in MODEL_EXTS
                or f.name.lower().endswith(".csv")
            ]

            for zf in zip_files:
                err = restore_session(zf.read())
                if err:
                    ss["_model_error"] = err
                else:
                    ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
                    st.rerun()

            if image_files:
                ss["skipped_files"] = process_uploads(image_files, mask_suffix) or []

            for model_file in model_files:
                if model_file.name.lower().endswith(".csv"):
                    # treat as Cellpose hyperparameter results
                    try:
                        hp_df = pd.read_csv(model_file)
                        required_cols = {
                            "cellprob",
                            "flow_threshold",
                            "niter",
                            "min_size",
                        }
                        if required_cols.issubset(hp_df.columns):
                            best = hp_df.iloc[0]
                            ss["cp_cellprob_threshold"] = float(best["cellprob"])
                            ss["cp_flow_threshold"] = float(best["flow_threshold"])
                            ss["cp_min_size"] = int(best["min_size"])
                            ss["cp_niter"] = int(best["niter"])
                            ss["cp_grid_results_df"] = hp_df
                            ss["_model_toast"] = (
                                f"HP set: cellprob={best['cellprob']:.2f}, "
                                f"flow={best['flow_threshold']:.2f}, "
                                f"min_size={int(best['min_size'])}, "
                                f"niter={int(best['niter'])}"
                            )
                        else:
                            missing = required_cols - set(hp_df.columns)
                            ss["_model_error"] = (
                                f"Missing columns: {', '.join(missing)}"
                            )
                    except Exception as e:
                        ss["_model_error"] = f"Failed to load HP CSV: {e}"
                else:
                    # auto-detect Cellpose vs DenseNet from state dict keys
                    data = model_file.read()
                    ext = os.path.splitext(model_file.name)[1].lower() or ".pth"
                    h = hashlib.sha1(data).hexdigest()[:12]
                    path = os.path.join(tempfile.gettempdir(), f"model_{h}{ext}")
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            f.write(data)
                    try:
                        state_dict = torch.load(path, map_location="cpu")
                        if "features.conv0.weight" in state_dict:
                            # DenseNet-121
                            num_classes = 2
                            if "classifier.2.weight" in state_dict:
                                num_classes = state_dict["classifier.2.weight"].shape[0]
                            elif "classifier.weight" in state_dict:
                                num_classes = state_dict["classifier.weight"].shape[0]
                            model = build_densenet(num_classes=num_classes)
                            model.load_state_dict(state_dict)
                            model.eval()
                            ss["densenet_model"] = model
                            ss["densenet_model_path"] = path
                            ss["densenet_ckpt_name"] = model_file.name
                            ss["_model_toast"] = (
                                f"Loaded DenseNet model: {model_file.name}"
                            )
                        elif any(k.startswith("downsample.") for k in state_dict):
                            # Cellpose UNet
                            ss["cellpose_model_bytes"] = data
                            ss["cellpose_model_name"] = model_file.name
                            ss["_model_toast"] = (
                                f"Loaded Cellpose model: {model_file.name}"
                            )
                        else:
                            ss["_model_error"] = (
                                f"Could not identify model type for: {model_file.name}"
                            )
                    except Exception as e:
                        ss["_model_error"] = f"Failed to load model: {e}"

            ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
            st.rerun()

        if toast_msg := ss.pop("_model_toast", None):
            st.toast(toast_msg, duration="infinite")
        if error_msg := ss.pop("_model_error", None):
            st.error(error_msg)

        # status
        st.divider()
        status_col2, status_col3 = st.columns(2)

        def _model_status_box(label: str, name: str | None) -> None:
            if name:
                st.markdown(
                    f"""<div style="background:rgba(33,195,84,0.12);border-left:4px solid #21c354;border-radius:0 8px 8px 0;padding:14px 18px;">
                    <p style="margin:0;font-size:1rem;color:#21c354;font-weight:600;letter-spacing:0.05em;">MODEL LOADED</p>
                    <p style="margin:4px 0 0;font-size:1.5rem;font-weight:700;">{label}</p>
                    <p style="margin:2px 0 0;font-size:1.1rem;opacity:0.75;word-break:break-all;">{name}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""<div style="background:rgba(255,135,0,0.12);border-left:4px solid #ff8700;border-radius:0 8px 8px 0;padding:14px 18px;">
                    <p style="margin:0;font-size:1rem;color:#ff8700;font-weight:600;letter-spacing:0.05em;">NOT UPLOADED</p>
                    <p style="margin:4px 0 0;font-size:1.5rem;font-weight:700;">{label}</p>
                    <p style="margin:2px 0 0;font-size:1.1rem;opacity:0.75;">Optional upload</p>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with status_col2:
            _model_status_box("Cellpose", ss.get("cellpose_model_name"))
        with status_col3:
            _model_status_box("DenseNet", ss.get("densenet_ckpt_name"))
        st.write("")

        # ---- Summary table: image–mask pairs ----

        ok = ordered_keys()
        if not ok:
            st.info("No images uploaded yet.")
        else:
            render_images_form()
