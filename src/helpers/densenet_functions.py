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
import subprocess
from pathlib import Path
import sys
import time
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ---- bring in existing app helpers ----
from src.helpers.state_ops import (
    ordered_keys,
    normalize_image,
    add_plotly_as_png_to_zip,
    plot_loss_curve,
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
    crop = resize_with_aspect_ratio(crop, patch_size=patch_size)
    return crop.astype(np.float32)


def resize_with_aspect_ratio(img: np.ndarray, patch_size=64) -> np.ndarray:
    """resizes input image to a square with 'patch_size' height while maintaining the aspect ratio"""
    th, tw = patch_size, patch_size
    h, w = img.shape[:2]

    # resize with aspect ratio
    scale = min(th / h, tw / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    )

    # pad to target size
    if img.ndim == 2:
        canvas = np.zeros((th, tw), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    else:
        c = img.shape[2]
        canvas = np.zeros((th, tw, c), dtype=img.dtype)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
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

    patches_np = [normalize_image(patch) for patch in patches]

    X_list = []
    for p in patches_np:
        p_chw = np.transpose(p, (2, 0, 1))
        X_list.append(torch.tensor(p_chw, dtype=torch.float32))

    if not X_list:
        return

    X_batch = torch.stack(X_list).to(device)

    with torch.no_grad():
        outputs = model(X_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    class_map = ensure_densenet_class_map()
    all_classes = ss.setdefault("all_classes", ["No label"])
    labels = rec.setdefault("labels", {})

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
        img = self.X[idx]
        img = np.transpose(img, (2, 0, 1))  # (3, 64, 64)

        tensor = torch.tensor(img, dtype=torch.float32)

        if tensor.max() > 1.0:
            tensor = tensor / 255.0

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return tensor, label


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
            patch = normalize_image(patch)

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


HERE = Path(__file__).resolve().parent
DENSENET_WORKER_SCRIPT = str(
    (HERE.parent / "training" / "densenet_worker.py").resolve()
)


def start_densenet_training(input_size, batch_size, epochs, val_split):
    """Starts DenseNet training asynchronously using a worker subprocess"""

    X, y, classes = load_labeled_patches(patch_size=input_size)
    if X.shape[0] < 2 or len(np.unique(y)) < 2:
        st.warning("Need at least 2 samples and 2 classes. Add more labeled cells.")
        return None

    tmpdir = tempfile.mkdtemp(prefix="densenet_train_")
    in_path = Path(tmpdir) / "input.npz"
    out_path = Path(tmpdir) / "output.npz"
    log_path = Path(tmpdir) / "training.log"

    np.savez_compressed(
        in_path,
        X=X,
        y=y,
        classes=classes,
        batch_size=batch_size,
        epochs=epochs,
        val_split=val_split,
    )

    cmd = [sys.executable, DENSENET_WORKER_SCRIPT, str(in_path), str(out_path)]

    log_file = open(log_path, "w")

    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    st.session_state["dn_training_job"] = {
        "process": process,
        "log_file": log_file,
        "tmpdir": tmpdir,
        "in_path": str(in_path),
        "out_path": str(out_path),
        "log_path": str(log_path),
        "input_size": input_size,
        "num_samples": X.shape[0],
        "classes": classes,
        "status": "running",
    }


def check_densenet_training_status():
    ss = st.session_state
    job = ss.get("dn_training_job")

    if not job:
        return None

    process = job["process"]
    log_path = Path(job["log_path"])
    returncode = process.poll()

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content:
                    job["log_content"] = content
        except Exception:
            # TODO: handle better
            pass

    if returncode is None:
        return "running"

    if "log_file" in job:
        try:
            job["log_file"].flush()
            job["log_file"].close()
        except:
            pass

    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                job["log_content"] = f.read()
        except:
            pass

    out_path = Path(job["out_path"])
    tmpdir = job["tmpdir"]

    try:
        if returncode != 0:
            job["status"] = "failed"
            job["error"] = (
                f"Training failed with exit code {returncode}\n\nLog:\n{job.get('log_content', 'No log available')}"
            )
            return "failed"

        if not out_path.exists():
            job["status"] = "failed"
            job["error"] = (
                f"Training failed: Output file not found.\n\nLog:\n{job.get('log_content', 'No log available')}"
            )
            return "failed"

        with np.load(out_path, allow_pickle=True) as data:
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

        job["status"] = "complete"
        job["history"] = history
        job["val_loader"] = None  # TODO: FIX can't serialize DataLoader
        job["classes"] = classes

        return "complete"

    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def cancel_densenet_training():
    ss = st.session_state
    job = ss.get("dn_training_job")

    if not job:
        return

    process = job["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    tmpdir = job["tmpdir"]
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)

    job["status"] = "cancelled"


def finetune_densenet(input_size, batch_size, epochs, val_split):
    """
    Legacy synchronous wrapper for backward compatibility.
    For async training, use start_densenet_training() instead.
    """
    start_densenet_training(input_size, batch_size, epochs, val_split)

    while True:
        status = check_densenet_training_status()
        if status == "complete":
            job = st.session_state["dn_training_job"]
            return job["history"], job["val_loader"], job["classes"]
        elif status == "failed":
            job = st.session_state["dn_training_job"]
            st.error("DenseNet training failed.")
            with st.expander("Show Error Details"):
                st.code(job["error"])
            return None, None, []
        time.sleep(1)


def evaluate_fine_tuned_densenet(history, val_loader, classes):
    model = st.session_state.get("densenet_model")
    if not model or not val_loader:
        return

    device = get_device()
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    st.info(
        "Model stored in session. You can use it immediately from the **Classify cells** panel."
    )

    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    train_losses = history.get("loss", [])
    val_losses = history.get("val_loss", [])
    ss["densenet_training_losses"] = plot_loss_curve(train_losses, val_losses)

    metrics = {
        "Accuracy": acc,
        "Precision": prec_m,
        "Recall": rec_m,
        "F1": f1_m,
    }
    ss["densenet_training_metrics"] = plot_densenet_metrics(metrics)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    st.session_state["densenet_confusion_matrix"] = plot_confusion_matrix(cm, classes)


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
    ok = ordered_keys()
    [st.session_state["images"][k]["name"].rsplit(".", 1)[0] for k in ok]

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
        params = dict(
            input_size=int(psize),
            epochs=int(ss.get("dn_max_epoch")),
            batch_size=int(ss.get("dn_batch_size")),
            val_split=float(ss.get("dn_val_split")),
            patches=len(labels),
            classes=labels["label"].nunique(),
        )

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
