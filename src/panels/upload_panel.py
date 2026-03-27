import streamlit as st
from src.helpers.state_ops import (
    ordered_keys,
)
from src.helpers.upload_download_functions import (
    process_uploads,
    render_images_form,
    load_demo_data,
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

    # ---------- Layout: 3 columns ----------
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # ---- single uploader: images & masks ----

        with st.container(border=True, height=420):
            st.subheader("Upload images & masks")
            st.caption(
                "Upload microscopy images (.tif, .png, .jpg) with optional paired mask files (.npy, .tif). Masks must share the image filename plus a suffix (default: _masks)."
            )

            up_key = f"u_all_np_{ss.get('uploader_nonce', 0)}"
            files = st.file_uploader(
                " ",
                type=["tif", "tiff", "npy", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=up_key,
                help="Unrecognised mask formats or extensions or masks without a paired image will be ignored.",
            )

            # allow user to specify mask suffix and toggle resizing
            suffix_col, resize_col = st.columns([2, 1], vertical_alignment="center")
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
                    help="Resize images & masks **while uploading** to 512x512. Recommended for large images, "
                    "but may cause quality loss. Calculated cell properties will still be correct for the "
                    "original (pre-resize) image.",
                )
                ss["resize_on_upload"] = resize_on_upload

            if files:
                ss["skipped_files"] = process_uploads(files, mask_suffix) or []
                ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
                st.rerun()

            if st.button("Use demo data", type="primary", width="stretch"):
                load_demo_data()

    with col2:
        with st.container(border=True, height=420):
            # ---- Cellpose model + optional hyperparameters ----
            st.subheader("Upload Cellpose segmenter and hyperparaeters")
            st.caption(
                "Upload a fine-tuned Cellpose model (.pt/.pth) trained in mycol. Optionally include a hyperparameter CSV to apply the best-found cellprob, flow threshold, niter, and min size."
            )
            cp_key = f"upload_cellpose_model_{ss.get('cp_uploader_nonce', 0)}"
            cp_files = st.file_uploader(
                " ",
                type=["pt", "pth", "csv"],
                accept_multiple_files=True,
                key=cp_key,
                help="Upload a Cellpose model (.pt/.pth) and optionally a hyperparameter CSV. "
                "The top row of the CSV will be applied.",
            )
            if cp_files:
                for cp_file in cp_files:
                    if cp_file.name.lower().endswith(".csv"):
                        # treat as hyperparameter results
                        try:
                            hp_df = pd.read_csv(cp_file)
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
                                ss["_cp_toast"] = (
                                    f"HP set: cellprob={best['cellprob']:.2f}, "
                                    f"flow={best['flow_threshold']:.2f}, "
                                    f"min_size={int(best['min_size'])}, "
                                    f"niter={int(best['niter'])}"
                                )
                            else:
                                missing = required_cols - set(hp_df.columns)
                                ss["_cp_error"] = (
                                    f"Missing columns: {', '.join(missing)}"
                                )
                        except Exception as e:
                            ss["_cp_error"] = f"Failed to load HP CSV: {e}"
                    else:
                        # treat as Cellpose model weights
                        ss["cellpose_model_bytes"] = cp_file.read()
                        ss["cellpose_model_name"] = cp_file.name
                ss["cp_uploader_nonce"] = ss.get("cp_uploader_nonce", 0) + 1
                st.rerun()

            if toast_msg := ss.pop("_cp_toast", None):
                st.toast(toast_msg, duration="infinite")
            if error_msg := ss.pop("_cp_error", None):
                st.error(error_msg)

            # display the currently loaded model
            cellpose_model = ss.get("cellpose_model_name") or "—"
            st.info(f"Loaded model: {cellpose_model}")

            # button to remove the currently loaded model
            if st.button("Clear Cellpose model", width="stretch"):
                ss["cellpose_model_bytes"] = None
                ss["cellpose_model_name"] = None
                ss["train_losses"] = []
                ss["test_losses"] = []

    with col3:
        # ---- DenseNet-121 classifier ----
        with st.container(border=True, height=420):
            st.subheader("Upload Densenet classifier")
            st.caption(
                "Upload a fine-tuned DenseNet-121 modle (.pt/.pth) trained in Mycol to classify segmented cells."
            )
            densenet_file = st.file_uploader(
                " ",
                type=["pth", "pt"],  # PyTorch formats instead
                key="upload_densenet_ckpt",
                help="Uploading a Densenet121 model is optional.",
            )
            if densenet_file is not None:
                data = densenet_file.read()
                ext = os.path.splitext(densenet_file.name)[1].lower() or ".pth"
                h = hashlib.sha1(data).hexdigest()[:12]
                path = os.path.join(tempfile.gettempdir(), f"densenet_{h}{ext}")
                if not os.path.exists(path):
                    with open(path, "wb") as f:
                        f.write(data)

                import torch
                from src.helpers.densenet_functions import build_densenet

                try:
                    state_dict = torch.load(path, map_location="cpu")
                    num_classes = 2  # TODO FIX
                    if "classifier.2.weight" in state_dict:
                        num_classes = state_dict["classifier.2.weight"].shape[0]

                    model = build_densenet(num_classes=num_classes)
                    model.load_state_dict(state_dict)
                    model.eval()

                except Exception as e:
                    st.error(f"Failed to load PyTorch model: {e}")
                    model = None

                ss["densenet_model"] = model
                ss["densenet_model_path"] = path
                ss["densenet_ckpt_name"] = densenet_file.name

            # display the currently loaded model
            densenet_model = ss.get("densenet_ckpt_name") or "—"
            st.info(f"Loaded model: {densenet_model}")

            # button to remove the currently loaded model
            if st.button("Clear DenseNet-121 model", width="stretch"):
                ss["densenet_model"] = None
                ss["densenet_ckpt_name"] = None

    # ---- Status panel ----
    st.divider()

    # ---- Summary table: image–mask pairs ----
    num_images = len(st.session_state["images"].keys())
    st.subheader(f"Images and Masks ({num_images})")

    ok = ordered_keys()
    if not ok:
        st.info("No images uploaded yet.")
    else:
        render_images_form()
