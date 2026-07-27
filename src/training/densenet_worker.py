import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from pathlib import Path


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ImageNet normalisation the pretrained backbone expects, applied AFTER augmentation
# (ColorJitter clamps float tensors to [0,1], so augment must run on [0,1] first).
_W = models.DenseNet121_Weights.IMAGENET1K_V1.transforms()
_NORM = transforms.Normalize(_W.mean, _W.std)

_TRAIN_TF = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        _NORM,
    ]
)
_VAL_TF = _NORM


class CellDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # X is already preprocessed app-side (patch_to_tensor): CHW float32 in [0, 1].
        # The worker only applies random augmentation on top.
        tensor = torch.tensor(self.X[idx], dtype=torch.float32)

        if self.transform:
            tensor = self.transform(tensor)

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return tensor, label


def build_densenet(num_classes=2):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, num_classes)
    )
    return model


def train_densenet(X, y, classes, batch_size, epochs, val_split):
    """Train DenseNet model and return history and best model state"""
    device = get_device()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, stratify=y, random_state=42
    )

    train_ds = CellDataset(X_train, y_train, transform=_TRAIN_TF)
    val_ds = CellDataset(X_val, y_val, transform=_VAL_TF)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_densenet(num_classes=len(classes))
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0
    best_model_state = None

    print(f"Starting DenseNet training for {epochs} epochs...")
    sys.stdout.flush()

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_ds)
        history["loss"].append(epoch_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_ds)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Loss: {epoch_val_loss:.4f}"
        )
        sys.stdout.flush()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  -> Validation loss improved to {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        sys.stdout.flush()

    if best_model_state:
        model.load_state_dict(best_model_state)

    return history, model.state_dict(), val_loader


import traceback


def main():
    try:
        if len(sys.argv) != 3:
            print("Usage: densenet_worker.py <input.npz> <output.npz>", file=sys.stderr)
            sys.exit(1)

        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])

        with np.load(in_path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            classes = data["classes"]
            batch_size = int(data["batch_size"])
            epochs = int(data["epochs"])
            val_split = float(data["val_split"])

        history, model_state, val_loader = train_densenet(
            X, y, classes, batch_size, epochs, val_split
        )

        device = get_device()
        model = build_densenet(num_classes=len(classes))
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))

        np.savez_compressed(
            out_path,
            model_state=model_state,
            history=history,
            classes=classes,
            metrics={
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            },
            confusion_matrix=cm,
        )
        sys.stdout.flush()

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
