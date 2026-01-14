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

        with st.container(border=True, height=350):
            st.subheader("Upload images & masks")

            up_key = f"u_all_np_{ss.get('uploader_nonce', 0)}"
            files = st.file_uploader(
                " ",
                type=["tif", "tiff", "npy", "png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=up_key,
                help="Unrecognised mask formats or extensions or masks without a paired image will be ignored.",
            )

            # allow user to specify mask suffix (default "_masks")
            mask_suffix = st.text_input(
                "Mask file names must match uploaded image an name plus this suffix",
                value=ss.get("mask_suffix", "_masks"),
                key="mask_suffix_input",
            )
            ss["mask_suffix"] = mask_suffix.strip() or "_masks"

            if files:
                ss["skipped_files"] = process_uploads(files, mask_suffix) or []
                ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
                st.rerun()

            if st.button("Use demo data", type="primary"):
                load_demo_data()

    with col2:
        with st.container(border=True, height=350):
            # ---- Cellpose model (custom weights) ----
            st.subheader("Upload Cellpose model")
            cellpose_file = st.file_uploader(
                " ",
                # Cellpose custom models are commonly .npy; allow pt/pth just in case
                type=["pt", "pth"],
                key="upload_cellpose_model",
                help="Uploading a Cellpose model is optional.",
            )
            if cellpose_file is not None:
                ss["cellpose_model_bytes"] = cellpose_file.read()
                ss["cellpose_model_name"] = cellpose_file.name

            # display the currently loaded model
            cellpose_model = ss.get("cellpose_model_name") or "—"
            st.info(f"Loaded model: {cellpose_model}")

            # button to remove the currently loaded model
            if st.button("Clear Cellpose model", width='stretch'):
                ss["cellpose_model_bytes"] = None
                ss["cellpose_model_name"] = None
                ss["train_losses"] = []
                ss["test_losses"] = []

    with col3:
        # ---- DenseNet-121 classifier ----
        with st.container(border=True, height=350):
            st.subheader("Upload Densenet classifier")
            densenet_file = st.file_uploader(
                " ",
                type=[
                    "pth", "pt"
                ],  #PyTorch formats instead
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
                    num_classes = 2 #TODO FIX
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
            if st.button("Clear DenseNet-121 model", width='stretch'):
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
