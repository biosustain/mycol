# helpers/densenet_functions.py
import numpy as np
import streamlit as st
import cv2
from scipy import ndimage
import pandas as pd
from PIL import Image
import io
from zipfile import ZipFile, ZIP_DEFLATED

import torch
import torch.nn as nn
from torchvision import models

# ---- bring in existing app helpers ----
from src.helpers.state_ops import (
    selected_training_keys,
    normalize_image,
    add_plotly_as_png_to_zip,
    params_to_csv,
    snapshot_for_undo,
)
from src.helpers.plot_helpers import (
    plot_loss_curve,
    plot_confusion_matrix,
    plot_densenet_metrics,
)
from src.helpers.job_runner import (
    start_worker_job,
    check_worker_job_status,
    cancel_worker_job,
)

ss = st.session_state


# -------------------------------
#  Device Configuration
# -------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------------------
#  Preprocessing and loader functions
# -------------------------------


def generate_cell_patch(image: np.ndarray, mask: np.ndarray, patch_size: int = 64):
    """Crop the mask's bounding box out of `image` and return it as a square
    patch_size patch, background blacked out. `mask` must have at least one pixel set."""
    # extract bounding box crop
    im, m = np.asarray(image), np.asarray(mask, bool)

    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    # coerce to 3 channels; images enter the app as RGB (add_image converts on
    # upload), so a 3-channel crop is already RGB and must not be reordered
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    else:
        crop = crop[..., :3]

    # resize to patch size
    crop = resize_with_aspect_ratio(crop, patch_size)
    return crop.astype(np.float32)


def resize_with_aspect_ratio(
    arr: np.ndarray,
    target: int | tuple[int, int],
    *,
    mode: str = "image",
) -> np.ndarray:
    """Resize `arr` to `target` size, preserving aspect ratio with centered zero padding.

    target: int  -> output is (target, target)
            tuple -> output is (target_h, target_w)
    mode:   "image" -> cv2.INTER_AREA when downscaling, INTER_LINEAR when upscaling
            "label" -> cv2.INTER_NEAREST throughout (preserves integer label ids)
    """
    th, tw = (target, target) if isinstance(target, int) else target
    h, w = arr.shape[:2]
    if (h, w) == (th, tw):
        return arr

    scale = min(th / h, tw / w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    if mode == "label":
        interp = cv2.INTER_NEAREST
    else:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR

    resized = cv2.resize(arr, (nw, nh), interpolation=interp)

    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    if arr.ndim == 2:
        canvas = np.zeros((th, tw), dtype=arr.dtype)
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    else:
        canvas = np.zeros((th, tw, arr.shape[2]), dtype=arr.dtype)
        canvas[y0 : y0 + nh, x0 : x0 + nw, :] = resized

    return canvas


def generate_patches_with_ids(rec, patch_size=64):
    """returns list of cell patches and patch ids from input record"""
    M = rec.get("masks")
    # one pass over the label image yields the bounding-box slice for every id
    patches, keep_ids = [], []
    for iid, sl in enumerate(ndimage.find_objects(M), start=1):
        if sl is None:  # id absent (gap left by a merge/cut)
            continue
        patches.append(generate_cell_patch(rec["image"][sl], M[sl] == iid, patch_size))
        keep_ids.append(iid)

    return patches, keep_ids


# -------------------------------
#  Model Helper functions
# -------------------------------


def get_densenet_num_classes(model) -> int | None:
    """Infer number of output classes from the DenseNet model."""
    if model is None:
        return None
    try:
        if isinstance(model.classifier, nn.Sequential):
            last_layer = model.classifier[-1]
            return last_layer.out_features
        return model.classifier.out_features
    except Exception:
        return None


def _ensure_assignable_classes(n_classes: int) -> list[str]:
    """Return at least n non-'No label' classes, creating ClassN names as needed."""
    all_classes = ss.setdefault("all_classes", ["No label"])
    real_classes = [c for c in all_classes if c != "No label"]

    next_idx = 1
    while len(real_classes) < n_classes:
        name = f"Class{next_idx}"
        next_idx += 1
        if name in all_classes:
            continue
        all_classes.append(name)
        real_classes.append(name)

    ss["all_classes"] = all_classes
    return real_classes


def initialize_densenet_class_map() -> dict[int, str]:
    """Rebuild the DenseNet output-to-class map for a newly uploaded model."""
    n_classes = get_densenet_num_classes(ss.get("densenet_model"))
    if n_classes is None:
        ss["densenet_class_map"] = {}
        return {}

    real_classes = _ensure_assignable_classes(n_classes)
    class_map = {idx: real_classes[idx] for idx in range(n_classes)}
    ss["densenet_class_map"] = class_map
    return class_map


def ensure_densenet_class_map() -> dict[int, str | None]:
    """Ensure each model class index has a valid, non-'No label' mapping."""
    model = ss.get("densenet_model")
    n_classes = get_densenet_num_classes(model)
    if n_classes is None:
        return {}

    real_classes = _ensure_assignable_classes(n_classes)
    class_map = {
        idx: ss.get("densenet_class_map", {}).get(idx) for idx in range(n_classes)
    }

    for idx in range(n_classes):
        if class_map[idx] not in real_classes:
            class_map[idx] = real_classes[idx]

    ss["densenet_class_map"] = class_map
    return class_map


def densenet_mapping_fragment():
    model = ss.get("densenet_model")
    if model is None:
        return

    n_classes = get_densenet_num_classes(model)
    all_classes = ss.setdefault("all_classes", ["No label"])
    class_map = ensure_densenet_class_map()

    for idx in range(n_classes):
        current = class_map.get(idx)
        options = [c for c in all_classes if c != "No label"]
        if current in options:
            default_idx = options.index(current)
        else:
            default_idx = 0

        selected = st.selectbox(
            label=f"Map model class {idx+1} to",
            options=options,
            index=default_idx,
            key=f"densenet_map_{idx}",
        )
        class_map[idx] = selected

    ss["densenet_class_map"] = class_map


def classify_cells_with_densenet(rec: dict, *, snapshot: bool = True) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model.

    Pass ``snapshot=False`` for batch runs so they aren't recorded for undo."""
    model = ss.get("densenet_model")
    M = rec.get("masks")

    if not np.any(M) or model is None:
        return

    device = get_device()
    model.to(device)
    model.eval()

    patches, keep_ids = generate_patches_with_ids(rec)

    X_list = [patch_to_tensor(p) for p in patches]

    if not X_list:
        return

    X_batch = torch.stack(X_list).to(device)

    with torch.no_grad():
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    class_map = ensure_densenet_class_map()
    all_classes = ss.setdefault("all_classes", ["No label"])
    labels = rec.setdefault("labels", {})

    # snapshot so a single-image classify can be undone (batch passes snapshot=False)
    if snapshot:
        snapshot_for_undo(rec)

    for iid, cls_idx in zip(keep_ids, preds):
        idx = int(cls_idx)
        name = class_map.get(idx)
        if not name:
            name = "No label"
        labels[int(iid)] = name

        if name and name != "No label" and name not in all_classes:
            all_classes.append(name)

    ss["all_classes"] = all_classes


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """Convert an HWC patch to a normalised CHW float32 tensor in [0, 1].

    Must be applied identically by the training and inference pipelines.
    """
    patch = normalize_image(patch)  # scale mean → 127.5, clip to [0,255], uint8
    chw = np.transpose(patch, (2, 0, 1))  # HWC → CHW
    return torch.tensor(chw, dtype=torch.float32) / 255.0  # [0,255] → [0,1]


# -------------------------------
#  Densenet121 Training
# -------------------------------


def build_densenet(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, num_classes)
    )
    return model


def load_labeled_patches(patch_size: int = 64):
    """
    Build X, y from all loaded images with labels.
    """
    ims = ss.get("images", {}) or {}
    all_classes = [
        c for c in ss.get("all_classes", []) if c != "No label"
    ]
    if not all_classes:
        all_classes = ["class0", "class1"]
    name_to_idx = {c: i for i, c in enumerate(all_classes)}

    # restrict to the images chosen on the training page (all when no selection yet)
    X, y = [], []
    for k in selected_training_keys("dn"):
        rec = ims.get(k) or {}
        img, M, labs = rec.get("image"), rec.get("masks"), rec.get("labels", {})
        if img is None or not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
            continue
        for iid, sl in enumerate(ndimage.find_objects(M), start=1):
            if sl is None:
                continue
            cname = labs.get(iid)
            if not cname or cname == "No label":
                continue
            patch = generate_cell_patch(img[sl], M[sl] == iid, patch_size)
            X.append(patch)
            y.append(name_to_idx[cname])

    if not X:
        return (
            np.zeros((0, patch_size, patch_size, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            all_classes,
        )

    X = np.stack(X, axis=0)  # (N, H, W, 3)
    y = np.array(y, dtype=np.int64)
    return X, y, all_classes


def start_densenet_training(input_size, batch_size, epochs, val_split):
    """Starts DenseNet training asynchronously using a worker subprocess"""
    X, y, classes = load_labeled_patches(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return None

    X = np.stack([patch_to_tensor(x).numpy() for x in X])  # (N, 3, H, W) float32 in [0, 1]

    start_worker_job(
        job_key="dn_training_job",
        worker_name="densenet",
        inputs=dict(
            X=X,
            y=y,
            classes=classes,
            batch_size=batch_size,
            epochs=epochs,
            val_split=val_split,
        ),
        metadata={
            "input_size": input_size,
            "num_samples": X.shape[0],
            "classes": classes,
        },
    )


def check_densenet_training_status():
    def _on_complete(data, job):
        model_state = data["model_state"].item()
        history = data["history"].item()
        classes = data["classes"]
        metrics = data["metrics"].item()
        cm = data["confusion_matrix"]

        model = build_densenet(num_classes=len(classes))
        model.load_state_dict(model_state)

        ss["densenet_ckpt_name"] = "densenet_finetuned"
        ss["densenet_model"] = model

        train_losses = history["loss"]
        val_losses = history["val_loss"]
        ss["densenet_training_losses"] = plot_loss_curve(train_losses, val_losses)
        ss["densenet_training_metrics"] = plot_densenet_metrics(metrics)
        ss["densenet_confusion_matrix"] = plot_confusion_matrix(cm, classes)

        job["history"] = history
        job["val_loader"] = None  # TODO: FIX can't serialize DataLoader
        job["classes"] = classes

    return check_worker_job_status("dn_training_job", _on_complete)


def cancel_densenet_training():
    cancel_worker_job("dn_training_job")


# -------------------------------
#  Visualization Functions
# -------------------------------


def array_to_png_bytes(arr: np.ndarray) -> bytes:
    """Convert float/uint arrays to PNG bytes (3-channel)."""
    a = arr
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] > 3:
        a = a[:, :, :3]

    if a.dtype.kind == "f":
        a = np.clip(a, 0, 255)
        if a.max() <= 1.0:
            a = (a * 255.0).round()
    a = np.clip(a, 0, 255).astype(np.uint8)

    bio = io.BytesIO()
    Image.fromarray(a).save(bio, format="PNG")
    return bio.getvalue()


def _write_patchset(zf, patch_size: int = 64) -> int:
    """Write cell-patch PNGs and their labels CSV directly into an open ZipFile.

    Returns the number of patches written (0 when there are no labelled cells)."""
    X, y, classes = load_labeled_patches(patch_size=patch_size)
    if X.shape[0] == 0:
        return 0

    rows = []
    for i in range(X.shape[0]):
        fname = f"patch_{i+1:04d}.png"
        label_idx = int(y[i])
        label_name = classes[label_idx] if 0 <= label_idx < len(classes) else "unknown"

        zf.writestr(f"cell_patches/{fname}", array_to_png_bytes(X[i]))
        rows.append({"filename": fname, "label_idx": label_idx, "label": label_name})

    zf.writestr(
        "cell_patch_labels.csv",
        pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
    )
    return X.shape[0]


def write_densenet_param_csvs(zf, input_size=None) -> None:
    """Write the DenseNet training hyperparameters CSV into `zf`."""
    params = {
        "input_size": int(input_size) if input_size is not None else ss.get("dn_input_size", 64),
        "batch_size": ss.get("dn_batch_size", 32),
        "max_epoch": ss.get("dn_max_epoch", 100),
        "val_split": ss.get("dn_val_split", 0.2),
    }
    zf.writestr("densenet_training_hyperparameters.csv", params_to_csv(params))


def densenet_model_bytes(model) -> bytes:
    """Serialize a DenseNet model's state_dict to .pth bytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def build_densenet_zip_bytes(psize):
    """Assemble the DenseNet training ZIP from session state."""

    buf = io.BytesIO()
    with ZipFile(buf, "w", ZIP_DEFLATED) as zout:
        # patches are written into the final zip 
        if _write_patchset(zout, psize) == 0:
            return None

        if ss.get("densenet_model") is not None:
            zout.writestr("densenet_model.pth", densenet_model_bytes(ss["densenet_model"]))

        write_densenet_param_csvs(zout, input_size=psize)

        add_plotly_as_png_to_zip(
            "densenet_training_losses", zout, "plots/densenet_training_losses.png"
        )
        add_plotly_as_png_to_zip(
            "densenet_training_metrics", zout, "plots/densenet_performance_metrics.png"
        )
        add_plotly_as_png_to_zip(
            "densenet_confusion_matrix", zout, "plots/densenet_confusion.png"
        )

    return buf.getvalue()
