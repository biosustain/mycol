# helpers/image_io.py
from PIL import Image
import io
import json
import os
import tempfile
import hashlib
import numpy as np
import tifffile as tiff
import streamlit as st
from zipfile import ZipFile
from pathlib import Path
from zipfile import ZIP_DEFLATED
from PIL import ImageDraw
import pandas as pd
from PIL import UnidentifiedImageError
from src.helpers.classifying_functions import (
    classes_map_from_labels,
    create_colour_palette,
)
from src.helpers.mask_editing_functions import create_image_mask_overlay
from src.helpers.densenet_functions import (
    resize_with_aspect_ratio,
    generate_cell_patch,
    array_to_png_bytes,
)
from src.helpers.cell_metrics_functions import build_per_image_counts


from src.helpers.state_ops import (
    ordered_keys,
    stem,
    set_current_by_index,
    reset_global_state_defaults,
)

from src.helpers.state_ops import normalize_image

ss = st.session_state

# --------------------------------------
# ---------- UPLOAD FUNCTIONS ----------
# --------------------------------------


def load_demo_data():
    """Restore the bundled example_session.zip through the normal session-restore pipeline."""
    app_dir = Path(__file__).parent.parent.parent  # repo root (where app.py lives)
    session_path = app_dir / "example_session.zip"
    if not session_path.exists():
        st.error(f"Demo session file not found: {session_path}")
        return
    err = restore_session(session_path.read_bytes())
    if err:
        st.error(f"Could not load demo data: {err}")
    else:
        ss["uploader_nonce"] = ss.get("uploader_nonce", 0) + 1
        st.rerun()


def restore_session(zip_bytes: bytes) -> str | None:
    """Restore a saved session zip into session state. Returns an error string or None on success."""
    from src.helpers.state_ops import (
        reset_global_state_defaults,
        set_current_by_index,
        ordered_keys,
    )

    def _cast(v, t, default):
        try:
            return default if pd.isna(v) else t(v)
        except (ValueError, TypeError):
            return default

    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        st.session_state.clear()
        reset_global_state_defaults()

        # ── Images ───────────────────────────────────────────────────────────
        for entry in sorted(
            n for n in names if n.startswith("images/") and not n.endswith("/")
        ):
            fname = Path(entry).name
            img_np = np.array(
                Image.open(io.BytesIO(zf.read(entry))).convert("RGB"), dtype=np.uint8
            )
            H, W = img_np.shape[:2]
            k = ss["next_ord"]
            ss["next_ord"] += 1
            ss["images"][k] = {
                "name": fname,
                "id": k,
                "image": img_np,
                "H": H,
                "W": W,
                "orig_H": H,
                "orig_W": W,
                "masks": np.zeros((H, W), dtype=np.uint16),
                "labels": {},
                "boxes": [],
                "last_click_xy": None,
                "canvas": {"closed_json": None, "processed_count": 0},
            }
            ss["name_to_key"][fname] = k

        # ── Masks ─────────────────────────────────────────────────────────────
        stem_to_key = {Path(rec["name"]).stem: k for k, rec in ss["images"].items()}
        for entry in sorted(
            n for n in names if n.startswith("masks/") and not n.endswith("/")
        ):
            k = stem_to_key.get(Path(entry).stem)
            if k is None:
                continue
            rec = ss["images"][k]
            mask = tiff.imread(io.BytesIO(zf.read(entry))).astype(np.uint16)
            if mask.shape != (rec["H"], rec["W"]):
                mask = np.array(
                    Image.fromarray(mask).resize(
                        (rec["W"], rec["H"]), resample=Image.NEAREST
                    ),
                    dtype=np.uint16,
                )
            rec["masks"] = mask
            rec["labels"] = {int(i): None for i in np.unique(mask) if i != 0}

        # ── Per-image metadata (orig dimensions, boxes) ───────────────────────
        if "image_metadata.json" in names:
            image_metadata = json.loads(zf.read("image_metadata.json").decode())
            for k, rec in ss["images"].items():
                name = Path(rec["name"]).stem
                m = image_metadata.get(name, {})
                rec["orig_H"] = m.get("orig_H", rec["H"])
                rec["orig_W"] = m.get("orig_W", rec["W"])
                rec["boxes"] = m.get("boxes", [])
                rec["boxes_display"] = m.get("boxes_display", [])

        # ── Labels (from cell_metrics.csv) ────────────────────────────────────
        if "cell_metrics.csv" in names:
            df = pd.read_csv(io.StringIO(zf.read("cell_metrics.csv").decode()))
            if {"image", "mask #", "mask label"}.issubset(df.columns):
                all_labels = set()
                for _, row in df.iterrows():
                    k = stem_to_key.get(Path(str(row["image"])).stem)
                    if k is None:
                        continue
                    label = str(row["mask label"])
                    label = (
                        None if label in ("Unlabelled", "No label", "nan") else label
                    )
                    ss["images"][k]["labels"][int(row["mask #"])] = label
                    if label:
                        all_labels.add(label)
                if all_labels:
                    ss["all_classes"] = ["No label"] + sorted(all_labels)

        # ── Cellpose inference hyperparameters ────────────────────────────────
        if "cellpose_inference_hyperparameters.csv" in names:
            df = pd.read_csv(
                io.StringIO(zf.read("cellpose_inference_hyperparameters.csv").decode())
            )
            p = dict(zip(df["parameter"], df["value"]))
            ss["cp_ch1"] = _cast(p.get("channel_1"), int, 0)
            ss["cp_ch2"] = _cast(p.get("channel_2"), int, 0)
            ss["cp_diameter"] = _cast(p.get("diameter"), int, 0)
            ss["cp_cellprob_threshold"] = _cast(p.get("cellprob_threshold"), float, 0.0)
            ss["cp_flow_threshold"] = _cast(p.get("flow_threshold"), float, 0.0)
            ss["cp_min_size"] = _cast(p.get("min_size"), int, 0)
            ss["cp_niter"] = _cast(p.get("niter"), int, 500)

        # ── Cellpose training hyperparameters ─────────────────────────────────
        if "cellpose_training_hyperparameters.csv" in names:
            df = pd.read_csv(
                io.StringIO(zf.read("cellpose_training_hyperparameters.csv").decode())
            )
            p = dict(zip(df["parameter"], df["value"]))
            ss["cp_base_model"] = _cast(p.get("base_model"), str, "cyto3")
            ss["cp_max_epoch"] = _cast(p.get("max_epoch"), int, 100)
            ss["cp_learning_rate"] = _cast(p.get("learning_rate"), float, 0.01)
            ss["cp_weight_decay"] = _cast(p.get("weight_decay"), float, 0.0001)
            ss["cp_batch_size"] = _cast(p.get("batch_size"), int, 8)
            ss["cp_min_cells_per_image"] = _cast(p.get("min_cells_per_image"), int, 1)
            ss["cp_training_ch1"] = _cast(p.get("training_ch1"), int, 0)
            ss["cp_training_ch2"] = _cast(p.get("training_ch2"), int, 0)
            ss["cp_do_gridsearch"] = _cast(
                p.get("do_gridsearch"), lambda v: str(v).lower() == "true", False
            )
            ss["cp_n_trials"] = _cast(p.get("n_trials"), int, 20)

        # ── DenseNet training hyperparameters ─────────────────────────────────
        if "densenet_training_hyperparameters.csv" in names:
            df = pd.read_csv(
                io.StringIO(zf.read("densenet_training_hyperparameters.csv").decode())
            )
            p = dict(zip(df["parameter"], df["value"]))
            ss["dn_input_size"] = _cast(p.get("input_size"), int, 64)
            ss["dn_batch_size"] = _cast(p.get("batch_size"), int, 32)
            ss["dn_max_epoch"] = _cast(p.get("max_epoch"), int, 100)
            ss["dn_val_split"] = _cast(p.get("val_split"), float, 0.2)

        # ── Cellpose model ────────────────────────────────────────────────────
        if "cellpose_model.pt" in names:
            ss["cellpose_model_bytes"] = zf.read("cellpose_model.pt")
            ss["cellpose_model_name"] = "cellpose_model.pt"

        # ── DenseNet model ────────────────────────────────────────────────────
        if "densenet_model.pth" in names:
            import torch
            from src.helpers.densenet_functions import build_densenet

            data = zf.read("densenet_model.pth")
            path = os.path.join(
                tempfile.gettempdir(),
                f"model_{hashlib.sha1(data).hexdigest()[:12]}.pth",
            )
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(data)
            state_dict = torch.load(path, map_location="cpu")
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
            ss["densenet_ckpt_name"] = "densenet_model.pth"

        # ── DenseNet class map ────────────────────────────────────────────────
        if "densenet_class_map.csv" in names:
            df = pd.read_csv(io.StringIO(zf.read("densenet_class_map.csv").decode()))
            if {"class_index", "class_name"}.issubset(df.columns):
                ss["densenet_class_map"] = {
                    int(row["class_index"]): (
                        None if pd.isna(row["class_name"]) else str(row["class_name"])
                    )
                    for _, row in df.iterrows()
                }

        # ── Training plots ────────────────────────────────────────────────────
        import plotly.io as pio

        for filename, state_key in [
            ("cellpose_training_losses.json", "cellpose_training_losses"),
            ("cellpose_iou_comparison.json", "cellpose_iou_comparison"),
            (
                "cellpose_original_counts_comparison.json",
                "cellpose_original_counts_comparison",
            ),
            (
                "cellpose_tuned_counts_comparison.json",
                "cellpose_tuned_counts_comparison",
            ),
            ("densenet_training_losses.json", "densenet_training_losses"),
            ("densenet_training_metrics.json", "densenet_training_metrics"),
            ("densenet_confusion_matrix.json", "densenet_confusion_matrix"),
        ]:
            if filename in names:
                ss[state_key] = pio.from_json(zf.read(filename).decode())

        # ── Cellpose grid search results ──────────────────────────────────────
        if "cellpose_grid_search_results.csv" in names:
            ss["cp_grid_results_df"] = pd.read_csv(
                io.StringIO(zf.read("cellpose_grid_search_results.csv").decode())
            )

        ok = ordered_keys()
        if ok:
            set_current_by_index(len(ok) - 1)

    return None


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
        orig_h = rec.get("orig_H", H)
        orig_w = rec.get("orig_W", W)
        if (orig_h, orig_w) != (H, W):
            # mirror image upload geometry: aspect-ratio resize + centered padding
            mask = resize_with_aspect_ratio(mask, (H, W), mode="label")
        else:
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
        orig_h = rec.get("orig_H", H)
        orig_w = rec.get("orig_W", W)
        if (orig_h, orig_w) != (H, W):
            # mirror image upload geometry: aspect-ratio resize + centered padding
            mask = resize_with_aspect_ratio(mask, (H, W), mode="label")
        else:
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
        orig_H, orig_W = img_np.shape[:2]
        # optionally resize images to 512x512
        if st.session_state.get("resize_on_upload", True):
            img_np = resize_with_aspect_ratio(img_np, 512)
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
        "orig_H": orig_H,
        "orig_W": orig_W,
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
            width="stretch",
            column_config={"Remove": st.column_config.CheckboxColumn()},
            disabled=["Image", "Masks Present", "Number of Masks", "Number of Labels"],
        )
        # handle removals
        if st.form_submit_button("Remove selected images", width="stretch"):
            for k in edited.loc[edited["Remove"]].index:
                ss.images.pop(k, None)

            # simplest/cleanest: rebuild mapping from current records only
            ss.name_to_key = {
                rec.get("name"): key
                for key, rec in ss.images.items()
                if rec.get("name")
            }

            ks = sorted(ss.images)
            ss.current_key = ks[0] if ks else None
            st.rerun()


# --------------------------------------
# --------- DOWNLOAD FUNCTIONS ---------
# --------------------------------------


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
        counts_df, class_cols = build_per_image_counts(key_order)
        counts_by_key = {r["_key"]: r for r in counts_df.to_dict("records")}
        summary_rows = []

        # iterate through records
        for k in key_order:
            rec = state_images[k]
            name = Path(rec.get("name", f"{k}")).stem
            crow = counts_by_key.get(k, {})
            counts = {c: int(crow.get(c, 0)) for c in class_cols}

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
            row["total"] = int(crow.get("Total masks", 0))
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
                    # extract patch (background blacked out, square-padded) using
                    # the same routine as the DenseNet training pipeline so the
                    # downloaded patches match the ones used for training
                    if not np.any(mask == iid):
                        continue
                    patch = generate_cell_patch(image=rec["image"], mask=(mask == iid))
                    # write patch to zip
                    patch_filename = f"patches/{name}_id{iid}_{cname_str}.png"
                    zf.writestr(patch_filename, array_to_png_bytes(patch))

        if include_summary:

            # --- write summary.csv into the zip (image + per-class + total)
            df = pd.DataFrame(summary_rows, columns=["image"] + class_cols + ["total"])
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            zf.writestr("cell_counts_per_image.csv", csv_buf.getvalue())

    return buf.getvalue()
