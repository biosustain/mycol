# helpers/densenet_functions.py
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from PIL import Image
import io
from zipfile import ZipFile, ZIP_DEFLATED
import os
import tempfile
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms

# ---- bring in existing app helpers ----
from src.helpers.state_ops import (
    ordered_keys,
    normalize_image,
    add_plotly_as_png_to_zip,
    plot_loss_curve,
    snapshot_for_undo,
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
    """takes an image and boolean mask input and a normalized square patch image from the mask"""
    # extract bounding box crop
    im, m = np.asarray(image), np.asarray(mask, bool)

    # handle empty mask case
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop, mc = im[y0:y1, x0:x1], m[y0:y1, x0:x1]
    crop = (crop * mc[..., None] if crop.ndim == 3 else crop * mc).astype(im.dtype)

    # checks to make sure crop is the correct format
    if crop.ndim == 2:
        crop = np.stack([crop] * 3, axis=-1)
    elif crop.ndim == 3 and crop.shape[2] == 4:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
    elif crop.ndim == 3 and crop.shape[2] == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
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
    # extract the individual masks
    ids = [int(v) for v in np.unique(M) if v != 0]

    patches, keep_ids = [], []
    for iid in ids:
        patches.append(
            generate_cell_patch(
                image=rec["image"], mask=M == iid, patch_size=patch_size
            )
        )
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


def ensure_densenet_class_map() -> dict[int, str | None]:
    """Ensure we have a mapping for each model class index in session_state."""
    ss = st.session_state
    model = ss.get("densenet_model")
    n_classes = get_densenet_num_classes(model)
    if n_classes is None:
        return {}

    class_map = ss.setdefault("densenet_class_map", {})
    # Make sure there is a key for each model output index
    for idx in range(n_classes):
        class_map.setdefault(idx, None)
    ss["densenet_class_map"] = class_map
    return class_map


def densenet_mapping_fragment():
    ss = st.session_state
    model = ss.get("densenet_model")
    if model is None:
        return

    n_classes = get_densenet_num_classes(model)
    all_classes = ss.setdefault("all_classes", ["No label"])
    class_map = ensure_densenet_class_map()

    for idx in range(n_classes):
        current = class_map.get(idx)
        options = all_classes
        if current in options:
            default_idx = options.index(current)
        else:
            default_idx = options.index("No label") if "No label" in options else 0

        selected = st.selectbox(
            label=f"Map model class {idx+1} to",
            options=options,
            index=default_idx,
            key=f"densenet_map_{idx}",
        )
        class_map[idx] = selected

    ss["densenet_class_map"] = class_map


def classify_cells_with_densenet(rec: dict) -> None:
    """Classify segmented cell masks in `rec` using a DenseNet-121 model."""
    ss = st.session_state
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

    # snapshot before writing labels so this (and each image in a batch) can be undone
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


# -------------------------------
#  Augmentation & Transforms
# -------------------------------


def apply_random_augmentations(img_tensor):
    """
    Apply random transforms on a (3, H, W) tensor.
    Simple manual implementation to match previous logic logic or utilize Torchvision transforms.
    Here we use torchvision transforms for simplicity and speed.
    """
    t = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ]
    )
    return t(img_tensor)


class CellDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X  # expecting (N, H, W, C) numpy arrays 0..1 or 0..255
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        tensor = patch_to_tensor(self.X[idx])

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return tensor, label


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    """Convert a preprocessed HWC uint8 patch to a normalised CHW float32 tensor in [0, 1].

    This is the single authoritative preprocessing step that must be called
    identically by both the training pipeline and the inference pipeline.
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
    ims = st.session_state.get("images", {}) or {}
    all_classes = [
        c for c in st.session_state.get("all_classes", []) if c != "No label"
    ]
    if not all_classes:
        all_classes = ["class0", "class1"]
    name_to_idx = {c: i for i, c in enumerate(all_classes)}

    X, y = [], []
    for k in ordered_keys():
        rec = ims.get(k) or {}
        img, M, labs = rec.get("image"), rec.get("masks"), rec.get("labels", {})
        if img is None or not isinstance(M, np.ndarray) or M.ndim != 2 or not np.any(M):
            continue
        ids = [int(v) for v in np.unique(M) if v != 0]
        for iid in ids:
            cname = labs.get(int(iid))
            if not cname or cname == "No label":
                continue

            patch = generate_cell_patch(
                image=img, mask=(M == iid), patch_size=patch_size
            )

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

        ss = st.session_state
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


def plot_confusion_matrix(cm, class_names):
    n = len(class_names)
    text = [[f"{cm[i,j]}" for j in range(n)] for i in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            text=text,
            textfont=dict(size=20),
            texttemplate="%{text}",
            colorscale="Blues",
            hoverongaps=False,
            showscale=False,
        )
    )
    fig.update_layout(
        title="Class Confusion Matrix",
        xaxis=dict(title="Predicted Class", tickangle=45),
        yaxis=dict(title="True Class", autorange="reversed"),
        width=max(500, 80 * n),
        height=max(400, 80 * n),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=40, b=80),
    )
    return fig


def plot_densenet_metrics(metrics):
    labels, values = list(metrics.keys()), list(metrics.values())
    fig = go.Figure(layout=dict(barcornerradius=10))
    fig.add_bar(
        x=labels,
        y=values,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
        marker=dict(color=["#EBF1F8"] * 4, line=dict(color="#004280", width=2)),
        name="metrics",
    )
    fig.update_yaxes(range=[0, 1.2])
    fig.update_layout(
        title="Validation metrics",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        width=450,
    )
    return fig


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


def build_patchset_zip(patch_size: int = 64) -> bytes | None:
    X, y, classes = load_labeled_patches(patch_size=patch_size)
    if X.shape[0] == 0:
        return None

    buf, rows = io.BytesIO(), []

    with ZipFile(buf, "w", ZIP_DEFLATED) as zf:
        for i in range(X.shape[0]):
            fname = f"patch_{i+1:04d}.png"
            label_idx = int(y[i])
            label_name = (
                classes[label_idx] if 0 <= label_idx < len(classes) else "unknown"
            )

            zf.writestr(f"cell_patches/{fname}", array_to_png_bytes(X[i]))
            rows.append(
                {"filename": fname, "label_idx": label_idx, "label": label_name}
            )

        zf.writestr(
            "cell_patch_labels.csv",
            pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
        )
    return buf.getvalue()


def build_densenet_zip_bytes(psize):
    """Assemble the DenseNet training ZIP from session state."""
    ss = st.session_state
    pzip = build_patchset_zip(psize)
    if not pzip:
        return None

    with ZipFile(io.BytesIO(pzip)) as zin:
        labels = pd.read_csv(io.BytesIO(zin.read("cell_patch_labels.csv")))

        # an uploaded (vs trained) model has no training params set, so coerce
        # only when present and skip missing entries
        def _coerce(key, cast):
            val = ss.get(key)
            return cast(val) if val is not None else None

        params = {
            k: v
            for k, v in dict(
                input_size=int(psize) if psize is not None else None,
                epochs=_coerce("dn_max_epoch", int),
                batch_size=_coerce("dn_batch_size", int),
                val_split=_coerce("dn_val_split", float),
                patches=len(labels),
                classes=labels["label"].nunique(),
            ).items()
            if v is not None
        }

        buf = io.BytesIO()
        with ZipFile(buf, "w", ZIP_DEFLATED) as zout:
            if ss.get("densenet_model") is not None:
                tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
                tmp_path = tmp.name
                tmp.close()
                torch.save(ss["densenet_model"].state_dict(), tmp_path)
                with open(tmp_path, "rb") as f:
                    zout.writestr("densenet_finetuned.pth", f.read())
                os.remove(tmp_path)

            zout.writestr(
                "training_params.csv",
                pd.Series(params)
                .rename_axis("parameter")
                .reset_index(name="value")
                .to_csv(index=False),
            )
            for n in zin.namelist():
                zout.writestr(n, zin.read(n))

            add_plotly_as_png_to_zip(
                "densenet_training_losses", zout, "plots/densenet_training_losses.png"
            )
            add_plotly_as_png_to_zip(
                "densenet_training_metrics",
                zout,
                "plots/densenet_performance_metrics.png",
            )
            add_plotly_as_png_to_zip(
                "densenet_confusion_matrix", zout, "plots/densenet_confusion.png"
            )

    return buf.getvalue()
