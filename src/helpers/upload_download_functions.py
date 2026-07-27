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
from src.helpers.grid_helpers import render_image_selection_table


from src.helpers.state_ops import (
    ordered_keys,
    stem,
    set_current_by_index,
    reset_global_state_defaults,
    write_image_to_zip,
    write_mask_to_zip,
    SESSION_PLOT_KEYS,
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


def _cast(v, t, default):
    """Cast v to type t, falling back to default on NaN / failure."""
    try:
        return default if pd.isna(v) else t(v)
    except (ValueError, TypeError):
        return default


def apply_cellpose_inference_params(p: dict) -> bool:
    """Set the Cellpose inference hyperparameters from a {parameter: value} mapping.
    Returns True if the mapping contained inference keys."""
    if not {"diameter", "cellprob_threshold", "flow_threshold", "min_size", "niter"} & set(p):
        return False
    ss["cp_diameter"] = _cast(p.get("diameter"), int, 0)
    ss["cp_cellprob_threshold"] = _cast(p.get("cellprob_threshold"), float, 0.0)
    ss["cp_flow_threshold"] = _cast(p.get("flow_threshold"), float, 0.0)
    ss["cp_min_size"] = _cast(p.get("min_size"), int, 0)
    ss["cp_niter"] = _cast(p.get("niter"), int, 500)
    return True


def apply_hyperparameter_csv(df) -> str | None:
    """Apply a cellpose_inference_hyperparameters.csv (parameter/value table) to state.
    Returns a short summary of what was applied, or None if it is not a recognized
    inference-params table."""
    if {"parameter", "value"}.issubset(df.columns):
        p = dict(zip(df["parameter"], df["value"]))
        if apply_cellpose_inference_params(p):
            return (
                f"Inference params set: diameter={ss['cp_diameter']}, "
                f"cellprob={ss['cp_cellprob_threshold']:.2f}, "
                f"flow={ss['cp_flow_threshold']:.2f}, "
                f"min_size={ss['cp_min_size']}, niter={ss['cp_niter']}"
            )
    return None


def restore_session(zip_bytes: bytes) -> str | None:
    """Restore a saved session zip into session state. Returns an error string or None on success."""

    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = set(zf.namelist())
        ss.clear()
        reset_global_state_defaults()

        # ── Images ───────────────────────────────────────────────────────────
        # saved images are already the resized working image, so resize=False;
        # true orig dimensions are re-applied from image_metadata.json below
        for entry in sorted(
            n for n in names if n.startswith("images/") and not n.endswith("/")
        ):
            add_image(Path(entry).name, zf.read(entry), resize=False)

        # ── Per-image metadata (mask suffix, orig dimensions, boxes) ──────────
        mask_suffix, image_metadata = "_masks", {}
        if "image_metadata.json" in names:
            raw = json.loads(zf.read("image_metadata.json").decode())
            # new shape: {"mask_suffix": str, "images": {...}}; legacy: flat {...}
            if isinstance(raw.get("mask_suffix"), str):
                mask_suffix, image_metadata = raw["mask_suffix"], raw.get("images", {})
            else:
                image_metadata = raw
        ss["mask_suffix"] = mask_suffix
        for k, rec in ss["images"].items():
            name = Path(rec["name"]).stem
            m = image_metadata.get(name, {})
            rec["orig_H"] = m.get("orig_H", rec["H"])
            rec["orig_W"] = m.get("orig_W", rec["W"])
            rec["boxes"] = m.get("boxes", [])

        # ── Masks ─────────────────────────────────────────────────────────────
        for k, rec in ss["images"].items():
            img_stem = Path(rec["name"]).stem
            entry = f"masks/{img_stem}{mask_suffix}.tif"
            if entry not in names:
                entry = f"masks/{img_stem}.tif"  # legacy zips saved without the suffix
            if entry not in names:
                continue
            add_mask(rec, entry, zf.read(entry))

        # ── Labels (from cell_metrics.csv) ────────────────────────────────────
        # Version compatibility lives here: only these three columns are read, and
        # every descriptor is recomputed from the masks by build_analysis_df. So a
        # zip saved by any version loads, whatever descriptor columns it does or
        # does not carry. Keep this read restricted to labels.
        if "cell_metrics.csv" in names:
            stem_to_key = {Path(rec["name"]).stem: k for k, rec in ss["images"].items()}
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
            apply_cellpose_inference_params(dict(zip(df["parameter"], df["value"])))

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
            ss["cp_nimg_per_epoch"] = _cast(p.get("nimg_per_epoch"), int, None)
            ss["cp_min_cells_per_image"] = _cast(p.get("min_cells_per_image"), int, 1)
            ss["cp_do_gridsearch"] = _cast(
                p.get("do_gridsearch"), lambda v: str(v).lower() == "true", True
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

        # ── Models (Cellpose / DenseNet) ──────────────────────────────────────
        for model_entry in ("cellpose_model.pt", "densenet_model.pth"):
            if model_entry in names:
                try:
                    load_model(model_entry, zf.read(model_entry))
                except Exception:
                    pass

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

        for fig_key in SESSION_PLOT_KEYS:
            filename = f"{fig_key}.json"
            if filename in names:
                ss[fig_key] = pio.from_json(zf.read(filename).decode())

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
    resize = ss.get("resize_on_upload", True)

    # process images
    for f in imgs:
        try:
            add_image(f.name, f.read(), resize)
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
            try:
                add_mask(ss.images[k], f.name, f.read())
            except Exception:
                skipped.append(f.name)

    return skipped


# --------------------------------------
# ------ SHARED LOADING PIPELINE -------
# --------------------------------------
# add_image / add_mask / load_model are the single path through which images,
# masks and models enter the app, used by both the uploads page and session
# restore. Each takes a filename and raw bytes so the caller's source (an
# uploaded file or a zip entry) is irrelevant.


def add_image(name: str, data: bytes, resize: bool) -> None:
    """Create a new image record in state from raw image bytes."""
    m = ss.name_to_key
    if name in m:
        ss.current_key = m[name]
        return

    img_np = np.array(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)
    orig_H, orig_W = img_np.shape[:2]
    if resize:
        img_np = resize_with_aspect_ratio(img_np, 512)
    H, W = img_np.shape[:2]

    k = ss.next_ord
    ss.next_ord += 1
    ss.images[k] = {
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
    ss.current_key = k


def add_mask(rec: dict, name: str, data: bytes) -> None:
    """Decode a mask (.npy or .tif) from bytes, fit it to the record and set labels."""
    if name.lower().endswith(".npy"):
        mask = load_npy_mask(data, rec)
    else:
        mask = load_tif_mask(data, rec)
    rec["masks"] = mask
    rec["labels"] = {int(i): None for i in np.unique(mask) if i != 0}


def _fit_mask_to_record(mask, rec):
    """Resize a label matrix to the record's working dims, mirroring image geometry."""
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


def load_npy_mask(data, rec):
    """Read Cellpose *_seg.npy bytes and return a (H,W) label matrix fitted to rec."""
    arr = np.load(io.BytesIO(data), allow_pickle=True).item()
    # Cellpose stores masks in dict under 'masks'
    mask = arr["masks"].astype(np.uint16)
    return _fit_mask_to_record(mask, rec)


def load_tif_mask(data, rec):
    """Read label TIFF bytes and return a (H,W) label matrix fitted to rec."""
    mask = tiff.imread(io.BytesIO(data)).astype(np.uint16)
    return _fit_mask_to_record(mask, rec)


def load_model(name: str, data: bytes) -> str:
    """Detect and load a Cellpose or DenseNet checkpoint from bytes into state.

    Returns a status message; raises ValueError if the type is unrecognised.
    """
    import torch
    from src.helpers.densenet_functions import (
        build_densenet,
        initialize_densenet_class_map,
    )

    ext = os.path.splitext(name)[1].lower() or ".pth"
    path = os.path.join(
        tempfile.gettempdir(), f"model_{hashlib.sha1(data).hexdigest()[:12]}{ext}"
    )
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
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
        ss["densenet_ckpt_name"] = name
        initialize_densenet_class_map()
        return f"Loaded DenseNet model: {name}"

    if any(k.startswith("downsample.") for k in state_dict):
        # Cellpose UNet
        ss["cellpose_model_bytes"] = data
        ss["cellpose_model_name"] = name
        return (
            f"Loaded Cellpose model: {name}. Remember to upload the hyperparameters csv (from training) or set them on "
            "the Annotate Images page for high prediction accuracy."
        )

    raise ValueError(f"Could not identify model type for: {name}")


def render_images_form():
    """display the uploaded images table with a remove-selected action"""

    # tick rows to remove
    selected_keys = render_image_selection_table("uploads_remove", default_all=False)

    if st.button(
        "Remove selected images",
        width="stretch",
        disabled=not selected_keys,
        type="primary",
    ):
        for k in selected_keys:
            ss.images.pop(k, None)

        # rebuild the name-to-key mapping from the remaining records
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
                ss.setdefault("all_classes", ["No label"])
            )
            if include_overlay
            else None
        )

        # --- prep columns & rows for summary.csv
        counts_df, class_cols = build_per_image_counts(key_order)
        counts_by_key = {r["_key"]: r for r in counts_df.to_dict("records")}
        summary_rows = []
        mask_suffix = ss["mask_suffix"]

        # iterate through records
        for k in key_order:
            rec = state_images[k]
            name = Path(rec.get("name", f"{k}")).stem
            crow = counts_by_key.get(k, {})
            counts = {c: int(crow.get(c, 0)) for c in class_cols}

            # write mask
            write_mask_to_zip(zf, name, rec.get("masks"), mask_suffix)

            # write image
            img = np.asarray(rec["image"])

            # optionally normalize image
            if ss["dl_normalize_download"]:
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
            write_image_to_zip(zf, name, img)

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
