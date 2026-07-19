import streamlit as st
from src.helpers.state_ops import (
    ordered_keys,
)
from src.helpers.upload_download_functions import (
    process_uploads,
    render_images_form,
    load_demo_data,
    restore_session,
    load_model,
    apply_hyperparameter_csv,
)
import os
import pandas as pd

ss = st.session_state


def render_main():


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
                    # Cellpose inference hyperparameters (parameter/value table,
                    # e.g. cellpose_inference_hyperparameters.csv)
                    try:
                        msg = apply_hyperparameter_csv(pd.read_csv(model_file))
                        if msg:
                            ss["_model_toast"] = msg
                        else:
                            ss["_model_error"] = (
                                "Unrecognized CSV. Expected a Cellpose inference "
                                "hyperparameters file (a parameter/value table)."
                            )
                    except Exception as e:
                        ss["_model_error"] = f"Failed to load HP CSV: {e}"
                else:
                    # auto-detect Cellpose vs DenseNet from state dict keys
                    try:
                        ss["_model_toast"] = load_model(
                            model_file.name, model_file.read()
                        )
                    except ValueError as e:
                        ss["_model_error"] = str(e)
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
