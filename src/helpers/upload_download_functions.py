# helpers/image_io.py
from PIL import Image
import io
import numpy as np
import tifffile as tiff
import streamlit as st
from zipfile import ZipFile
from pathlib import Path
from zipfile import ZIP_DEFLATED
from PIL import ImageDraw
import pandas as pd
from PIL import UnidentifiedImageError
import os
import tempfile
import hashlib
from src.helpers.classifying_functions import classes_map_from_labels, create_colour_palette
from src.helpers.mask_editing_functions import create_image_mask_overlay
from src.helpers.densenet_functions import resize_with_aspect_ratio


from src.helpers.state_ops import (
    ordered_keys,
    stem,
    set_current_by_index,
    reset_global_state,
)

from src.helpers.state_ops import ordered_keys, get_current_rec, normalize_image

ss = st.session_state

# --------------------------------------
# ---------- UPLOAD FUNCTIONS ----------
# --------------------------------------

from pathlib import Path
import io

ss = st.session_state


class DemoUploadedFile(io.BytesIO):
    """
    Mimics Streamlit's UploadedFile using only the basename of a file path.
    """

    def __init__(self, file_path: Path):
        self._path = Path(file_path)
        super().__init__(self._path.read_bytes())

    @property
    def name(self):
        return self._path.name


def load_demo_data():
    """Load demo images, masks, Cellpose model and Densenet model into session_state."""

    DEMO_MASK_SUFFIX = "_masks"

    reset_global_state()

    # ---------- locate demo_data folder ----------
    demo_root = Path("demo_data")  # project-relative, like intro_images
    images_dir = demo_root / "images"
    masks_dir = demo_root / "masks"

    # ---------- prepare image & mask 'uploaded' files ----------
    image_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    mask_exts = {".tif", ".tiff", ".npy"}

    file_objs = []

    # images
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() in image_exts:
            file_objs.append(DemoUploadedFile(p))  # behaves enough like st.UploadedFile

    # masks
    for p in sorted(masks_dir.iterdir()):
        if p.suffix.lower() in mask_exts:
            file_objs.append(DemoUploadedFile(p))

    # use the same suffix that matches demo masks
    ss["mask_suffix"] = DEMO_MASK_SUFFIX

    # process images + masks through existing pipeline
    skipped = process_uploads(file_objs, DEMO_MASK_SUFFIX) or []
    ss["skipped_files"] = skipped

    #TODO: FIX, INJECT DUMMY LABELS FOR DEMO TRAINING
    demo_classes = ["Demo Class A", "Demo Class B"]
    ss["all_classes"] = ["No label"] + demo_classes
    
    rng = np.random.default_rng(42)
    
    for k in ordered_keys():
        rec = ss.images[k]
        masks = rec.get("masks")
        if masks is None: continue
        
        cell_ids = [i for i in np.unique(masks) if i != 0]
        
        new_labels = {}
        for cid in cell_ids:
            lbl = demo_classes[cid % 2] 
            new_labels[cid] = lbl
            
        rec["labels"] = new_labels
    
    # close the file handles now that everything is loaded
    for f in file_objs:
        try:
            f.close()
        except Exception:
            pass

    # ---------- load Cellpose model from disk ----------
    cellpose_path = demo_root / "cellpose_model.pt"
    if cellpose_path.exists():
        ss["cellpose_model_bytes"] = cellpose_path.read_bytes()
        ss["cellpose_model_name"] = cellpose_path.name

    # ---------- load Densenet model from disk ----------
    densenet_path = demo_root / "densenet_demo.pth"
    if densenet_path.exists():
        import torch
        from src.helpers.densenet_functions import build_densenet

        try:
            state_dict = torch.load(densenet_path, map_location="cpu")
            
            num_classes = 2
            if "classifier.2.weight" in state_dict:
                num_classes = state_dict["classifier.2.weight"].shape[0]
            elif "classifier.weight" in state_dict:
                 num_classes = state_dict["classifier.weight"].shape[0]

            model = build_densenet(num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.eval()

            ss["densenet_model"] = model
            ss["densenet_model_path"] = str(densenet_path)
            ss["densenet_ckpt_name"] = densenet_path.name
        except Exception as e:
            st.warning(f"Could not load demo DenseNet model: {e}")

    # ---------- notify + refresh UI ----------
    st.toast("Demo data loaded.")
    st.rerun()


def process_uploads(files, mask_suffix):
    """Process uploaded files: add images and masks to state. Return list of skipped filenames."""

    # early exit if no uploaded files
    if not files:
        return []
    skipped = []

    # separate images and masks
    mask_suffix_len = len(mask_suffix)
    imgs = [(f) for f in files if not stem(f.name).endswith(mask_suffix)]
    masks = [f for f in files if stem(f.name).endswith(mask_suffix)]

    # process images
    for f in imgs:
        try:
            create_new_record_with_image(f)
        except (UnidentifiedImageError, Exception):
            skipped.append(f.name)

    ok = ordered_keys()
    if ok:
        set_current_by_index(len(ok) - 1)

    # process masks
    if masks and ss.images:
        stem_to_key = {stem(rec["name"]): k for k, rec in ss.images.items()}
        for f in masks:
            base = stem(f.name)[:-mask_suffix_len]
            k = stem_to_key.get(base)
            if k is None:
                skipped.append(f.name)
                continue
            rec = ss.images[k]
            rec["labels"] = {}
            try:
                if f.name.endswith(".npy"):
                    rec["masks"] = load_npy_mask(f, rec)
                else:
                    rec["masks"] = load_tif_mask(f, rec)
                rec["labels"] = {
                    int(i): None for i in np.unique(rec["masks"]) if i != 0
                }
            except Exception:
                skipped.append(f.name)
                continue

    return skipped


def load_npy_mask(file, rec):
    """Read Cellpose *_seg.npy and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    arr = np.load(io.BytesIO(file), allow_pickle=True).item()
    # Cellpose stores masks in dict under 'masks'
    mask = arr["masks"].astype(np.uint16)
    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        # resize if needed
        from PIL import Image

        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def load_tif_mask(file, rec):
    """Read a label TIFF and return a (H,W) label matrix with 0 background, 1..N instances."""
    file = file.read()
    mask = tiff.imread(io.BytesIO(file)).astype(np.uint16)

    H, W = rec["H"], rec["W"]
    if mask.shape != (H, W):
        mask = np.array(
            Image.fromarray(mask).resize((W, H), resample=Image.NEAREST),
            dtype=np.uint16,
        )
    return mask


def create_new_record_with_image(uploaded_file):
    """Create a new image record in state from uploaded file."""

    # get name mappings and images dict
    name = uploaded_file.name
    m = st.session_state.name_to_key
    imgs = st.session_state.images

    # check for existing name
    if name in m:
        st.session_state.current_key = m[name]
        return

    try:
        # load image and convert to RGB
        img_np = np.array(Image.open(uploaded_file).convert("RGB"), dtype=np.uint8)
        # images are always resized to 512x512
        img_np = resize_with_aspect_ratio(img_np, patch_size=512)
    except (UnidentifiedImageError, Exception):
        raise

    # get image dimensions
    H, W = img_np.shape[:2]
    # create new record
    k = st.session_state.next_ord
    st.session_state.next_ord += 1
    imgs[k] = {
        "name": name,
        "id": k,
        "image": img_np,
        "H": H,
        "W": W,
        "masks": np.zeros((H, W), dtype=np.uint16),
        "labels": {},
        "boxes": [],
        "last_click_xy": None,
        "canvas": {"closed_json": None, "processed_count": 0},
    }
    m[name] = k
    st.session_state.current_key = k


def render_images_form():
    """display the uploaded images table"""
    ss, ok = st.session_state, sorted(st.session_state.images)

    # helper to check if mask present
    def is_mask(m):
        return isinstance(m, np.ndarray) and m.any()

    rows = []
    for i, k in enumerate(ok, start=1):
        rec, m = ss.images[k], ss.images[k].get("masks")
        has = is_mask(m)
        n = int(len(np.unique(m)) - 1) if has else 0
        nl = sum(v is not None for v in rec.get("labels", {}).values())
        rows.append(
            {
                "No.": i,  # image id number
                "Image": rec.get("name", k),  # image filename
                "Mask Present": "✅" if has else "❌",  # whether mask is present
                "Number of Masks": n,  # number of masks
                "Labelled Masks": f"{nl}/{n}",  # number of labelled masks
                "Remove": False,  # checkbox to remove image
            }
        )

    # render the data editor
    with st.form("images_form"):
        edited = st.data_editor(
            pd.DataFrame(rows, index=ok),
            hide_index=True,
            height=580,
            width='stretch',
            column_config={"Remove": st.column_config.CheckboxColumn()},
            disabled=["Image", "Masks Present", "Number of Masks", "Number of Labels"],
        )
        # handle removals
        if st.form_submit_button("Remove selected images", width='stretch'):
            for k in edited.loc[edited["Remove"]].index:
                ss.images.pop(k, None)
            ks = sorted(ss.images)
            ss.current_key = ks[0] if ks else None
            st.rerun()


# --------------------------------------
# --------- DOWNLOAD FUNCTIONS ---------
# --------------------------------------


def counts_for_rec(rec) -> dict:
    """Return dict class_name -> count for one record, using rec['labels'] mapping."""
    labels = rec.get("labels", {}) or {}
    # Values in labels can be strings or ints or None; normalize to str
    by_class = {}
    for iid, cname in labels.items():
        cname = "No label" if cname in (None, "", -1) else str(cname)
        by_class[cname] = by_class.get(cname, 0) + 1
    return by_class


def build_masks_images_zip(
    state_images,
    key_order,
    include_overlay: bool,
    include_counts: bool,
    include_patches: bool,
    include_summary: bool,
) -> bytes:
    """Build a ZIP file with masks, images (optionally with overlays), and summary CSV.
    Return the ZIP file as bytes."""

    buf = io.BytesIO()
    with ZipFile(buf, mode="w", compression=ZIP_DEFLATED) as zf:
        # Class color palette (only if overlays requested)
        palette = (
            create_colour_palette(
                st.session_state.setdefault("all_classes", ["No label"])
            )
            if include_overlay
            else None
        )

        # --- prep columns & rows for summary.csv
        class_cols = [c for c in st.session_state["all_classes"]]
        summary_rows = []

        # iterate through records
        for k in key_order:
            rec = state_images[k]
            name = Path(rec.get("name", f"{k}")).stem
            counts = counts_for_rec(rec)

            # write mask
            mask = rec.get("masks")
            tbuf = io.BytesIO()
            tiff.imwrite(tbuf, mask.astype(np.uint16))
            mask_suffix = ss["mask_suffix"]
            zf.writestr(f"masks/{name}{mask_suffix}.tif", tbuf.getvalue())

            # write iamge
            img = np.asarray(rec["image"])

            # optionally normalize image
            if st.session_state["dl_normalize_download"]:
                img = normalize_image(img)

            # optional overlay (colored masks) for image
            if include_overlay:
                classes_map = classes_map_from_labels(
                    rec.get("masks"), rec.get("labels", {})
                )
                img = create_image_mask_overlay(
                    img, rec.get("masks"), classes_map, palette, alpha=0.35
                )

            # optionally annotate image with class counts
            if include_counts:
                lines = [f"{cls}: {cnt}" for cls, cnt in sorted(counts.items())]
                if lines:
                    txt = "\n".join(lines)
                    pil = Image.fromarray(img)
                    d = ImageDraw.Draw(pil)

                    # measure text height
                    _, _, _, th = d.multiline_textbbox((0, 0), txt)

                    # create space on top
                    new_img = Image.new(
                        "RGB", (pil.width, pil.height + th + 10), "white"
                    )
                    new_img.paste(pil, (0, th + 10))

                    # centered text
                    d = ImageDraw.Draw(new_img)
                    tw = d.multiline_textbbox((0, 0), txt)[2]
                    d.multiline_text(((new_img.width - tw) // 2, 5), txt, fill="black")

                    img = np.array(new_img)

            # capture a row for the CSV
            row = {"image": name}
            row.update({c: int(counts.get(c, 0)) for c in class_cols})
            row["total"] = int(sum(counts.values()))
            summary_rows.append(row)

            # write processed image to zip file
            ibuf = io.BytesIO()
            tiff.imwrite(ibuf, img, photometric="rgb", compression="deflate")
            zf.writestr(f"images/{name}.tif", ibuf.getvalue())

        # optionally write cell patches into zip
        if include_patches:
            for k in key_order:
                rec = state_images[k]
                name = Path(rec.get("name", f"{k}")).stem
                mask = rec.get("masks")
                labels = rec.get("labels", {})

                unique_ids = [i for i in np.unique(mask) if i != 0]
                for iid in unique_ids:
                    cname = labels.get(iid)
                    cname_str = (
                        "No_label"
                        if cname in (None, "", -1)
                        else str(cname).replace(" ", "_")
                    )
                    # extract patch
                    ys, xs = np.where(mask == iid)
                    if ys.size == 0 or xs.size == 0:
                        continue
                    y1, y2 = ys.min(), ys.max() + 1
                    x1, x2 = xs.min(), xs.max() + 1
                    patch = rec["image"][y1:y2, x1:x2]
                    # write patch to zip
                    pbuf = io.BytesIO()
                    tiff.imwrite(pbuf, patch, photometric="rgb", compression="deflate")
                    patch_filename = f"patches/{name}_id{iid}_{cname_str}.tif"
                    zf.writestr(patch_filename, pbuf.getvalue())

        if include_summary:

            # --- write summary.csv into the zip (image + per-class + total)
            df = pd.DataFrame(summary_rows, columns=["image"] + class_cols + ["total"])
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr("cell_counts_per_image.csv", csv_buf.getvalue())

    return buf.getvalue()
