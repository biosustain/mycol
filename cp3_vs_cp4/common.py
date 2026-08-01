"""Shared harness for the Cellpose 3 vs Cellpose 4 comparison.

Every model here is trained and tuned by mycol's own worker scripts, invoked through
the same npz interface the app uses, so the runs are faithful rather than
re-implemented. Cellpose 3's workers live on the `main` branch and Cellpose 4's on
`cp4`, so each model is driven from its own checkout with its own environment.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "cp3_vs_cp4" / "results"
WORK_DIR = REPO / "cp3_vs_cp4" / "_work"

# --- the two checkouts: same app, different Cellpose generation ----------------------
CHECKOUT = {
    "cp3": REPO.parent / "mycol-main-cp3",   # `main` worktree: cellpose 3.1.1
    "cp4": REPO,                             # this branch:     cellpose 4.2.1.1
}

# --- mycol's own defaults, read off each branch's fine_tune_panel --------------------
# cp3: main @ src/panels/fine_tune_panel.py   cp4: cp4 @ src/panels/fine_tune_panel.py
DEFAULTS = {
    "cp3": dict(base_model="cyto2", epochs=100, learning_rate=0.1, weight_decay=1e-4,
                batch_size=8, nimg_per_epoch=None, n_trials=20),
    "cp4": dict(base_model="cpsam_v2", epochs=100, learning_rate=1e-5, weight_decay=0.1,
                batch_size=1, nimg_per_epoch=8, n_trials=20),
}

DATASETS = {
    "cs1": REPO / "case_study_1" / "mycol_saved_session_CS1.zip",
    "bright_ecoli": REPO / "case_study_3" / "bright_ecoli" / "mycol_saved_session_ecoli_cfu.zip",
    "s_aureus": REPO / "case_study_3" / "s_aureus" / "mycol_saved_session (19).zip",
    "mcount": REPO / "case_study_3" / "mcount" / "mycol_saved_session_mcount50.zip",
}


# ---------------------------------------------------------------- session + split ---
def load_session(dataset: str) -> dict:
    """Read a mycol saved session and rebuild the exact train/test split it used.

    mycol splits with src/training/data_split.py -- train_test_split(range(n),
    test_size=0.2, random_state=42) over `ordered_keys()`, after dropping images with
    fewer than `min_cells_per_image` annotated cells. `ordered_keys()` is the image
    upload order, and image_metadata.json is written by iterating it, so its key order
    is the authoritative ordering (not necessarily alphabetical).
    """
    sys.path.insert(0, str(REPO / "src" / "training"))
    from data_split import split_train_test  # identical on both branches

    with zipfile.ZipFile(DATASETS[dataset]) as z:
        names_in_zip = set(z.namelist())
        meta = json.loads(z.read("image_metadata.json"))
        order = list(meta["images"]) if isinstance(meta, dict) and "images" in meta else list(meta)
        suffix = meta.get("mask_suffix", "") if isinstance(meta, dict) else ""

        def mask_name(n):
            cand = f"masks/{n}{suffix}.tif"
            return cand if cand in names_in_zip else f"masks/{n}.tif"

        # the session's own min_cells decides which images are eligible, so it must come
        # from the session for the split to match; everything else uses mycol's defaults
        rows = csv.reader(io.StringIO(z.read("cellpose_training_hyperparameters.csv").decode()))
        hp = {r[0]: r[1] for r in rows if len(r) >= 2}
        min_cells = int(float(hp["min_cells_per_image"]))

        images = {n: np.array(Image.open(io.BytesIO(z.read(f"images/{n}.tif"))).convert("RGB"),
                              dtype=np.uint8) for n in order}
        masks = {n: np.array(Image.open(io.BytesIO(z.read(mask_name(n))))).astype(np.uint16)
                 for n in order}

    counts = {n: int(np.unique(masks[n])[1:].size) for n in order}
    kept = [n for n in order if counts[n] >= min_cells]
    tr, te = split_train_test(len(kept))
    train, test = [kept[i] for i in tr], [kept[i] for i in te]
    return dict(dataset=dataset, images=images, masks=masks, counts=counts,
                min_cells=min_cells, kept=kept, train=train, test=test,
                shape=images[order[0]].shape[:2])


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """mycol's preprocess_for_cellpose: grayscale, then mean intensity scaled to ~127.5.

    Replicated here (rather than imported) because the worker environments deliberately
    do not carry the app's Streamlit dependencies. `verify_preprocess_matches_app`
    checks it against the real function.
    """
    import cv2

    img = rgb
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im = img.astype(np.float32)
    mean_val = float(im.mean())
    if mean_val <= 0:
        rng = float(im.max() - im.min())
        im = (im - im.min()) / rng * 255.0 if rng > 0 else im * 0.0
    else:
        im = im * (127.5 / mean_val)
    return np.clip(im, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ worker driving ---
def _worker(model: str) -> tuple[str, str]:
    root = CHECKOUT[model]
    return str(root / "src" / "training" / ".venv" / "bin" / "python"), \
           str(root / "src" / "training" / "unified_worker.py")


def _run(model: str, mode: str, payload: dict, tag: str) -> tuple[dict, float]:
    """Run one mycol worker on an npz payload; return (outputs, wall seconds)."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    fin, fout = WORK_DIR / f"{tag}_in.npz", WORK_DIR / f"{tag}_out.npz"
    np.savez_compressed(fin, **payload)
    py, worker = _worker(model)
    t0 = time.time()
    proc = subprocess.run([py, worker, mode, str(fin), str(fout)],
                          capture_output=True, text=True)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        tail = "\n".join(l for l in proc.stderr.splitlines() if "it/s" not in l)[-2000:]
        raise RuntimeError(f"{model} {mode} worker failed:\n{tail}")
    with np.load(fout, allow_pickle=True) as d:
        out = {k: d[k] for k in d.files}
    fin.unlink(missing_ok=True)
    return out, elapsed


def _obj(arrays) -> np.ndarray:
    return np.array([np.ascontiguousarray(a) for a in arrays], dtype=object)


def train_model(session: dict, model: str) -> dict:
    """Fine-tune with mycol's finetune worker at mycol's default settings.

    The worker re-derives the split internally from the images it is given, so the full
    eligible set is passed in exactly as the app does.
    """
    d = DEFAULTS[model]
    imgs = [preprocess(session["images"][n]) for n in session["kept"]]
    masks = [session["masks"][n] for n in session["kept"]]
    payload = dict(images=_obj(imgs), masks=_obj(masks), base_model=d["base_model"],
                   epochs=d["epochs"], learning_rate=d["learning_rate"],
                   weight_decay=d["weight_decay"], batch_size=d["batch_size"],
                   nimg_per_epoch=(d["nimg_per_epoch"] or 0),
                   min_train_masks=session["min_cells"])
    if model == "cp3":                      # main's worker still takes `channels`
        payload["channels"] = np.array([0, 0])
    out, secs = _run(model, "finetune", payload, f"{session['dataset']}_{model}_train")

    import torch
    ckpt = WORK_DIR / f"{session['dataset']}_{model}_model.pt"
    torch.save(out["state_dict"].item(), ckpt)
    return dict(model_path=str(ckpt), train_seconds=secs,
                train_losses=out["train_losses"].tolist(),
                test_losses=out["test_losses"].tolist())


def tune_model(session: dict, model: str, model_path: str) -> dict:
    """Run mycol's validation worker: Optuna over the app's default search space."""
    d = DEFAULTS[model]
    imgs = [preprocess(session["images"][n]) for n in session["kept"]]
    masks = [session["masks"][n] for n in session["kept"]]
    payload = dict(images=_obj(imgs), masks=_obj(masks),
                   image_names=np.array(session["kept"], dtype=object),
                   base_model=d["base_model"], tuned_model_path=model_path,
                   do_gridsearch=True, n_trials=d["n_trials"])
    if model == "cp3":
        payload["channels"] = np.array([0, 0])
    out, secs = _run(model, "validation", payload, f"{session['dataset']}_{model}_tune")
    best = out["best_params"].item()
    trials = out["optuna_results"]
    return dict(best_params=best, tune_seconds=secs,
                n_trials_run=len(trials.tolist()) if trials.shape else 0)


# ------------------------------------------------------------------------ metrics ---
def evaluate(session: dict, model: str, model_path: str, best_params: dict) -> dict:
    """Score the tuned model on the held-out test images."""
    from cellpose import models as cp_models, metrics as cp_metrics
    import torch

    gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    net = cp_models.CellposeModel(gpu=gpu, pretrained_model=model_path)
    kw = dict(diameter=best_params.get("diameter") or None,
              cellprob_threshold=float(best_params.get("cellprob", 0.0)),
              flow_threshold=float(best_params.get("flow_threshold", 0.4)),
              niter=int(best_params.get("niter", 200)),
              min_size=int(best_params.get("min_size", 15)))
    if model == "cp3":
        kw["channels"] = [0, 0]

    imgs = [preprocess(session["images"][n]) for n in session["test"]]
    gt = [session["masks"][n] for n in session["test"]]
    t0 = time.time()
    preds = []
    for i in range(0, len(imgs), 8):
        out = net.eval(imgs[i:i + 8], **kw)[0]
        preds += list(out)
    infer_seconds = time.time() - t0

    ap, tp, fp, fn = cp_metrics.average_precision(gt, preds, threshold=[0.5, 0.75, 0.9])
    true = np.array([int(np.unique(g)[1:].size) for g in gt], dtype=float)
    pred = np.array([int(np.unique(np.asarray(p))[1:].size) for p in preds], dtype=float)

    ss_res = float(((pred - true) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    return dict(
        images=session["test"], true_counts=true.tolist(), pred_counts=pred.tolist(),
        r2=1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        mae=float(np.abs(pred - true).mean()),
        mape=float((np.abs(pred - true) / np.where(true > 0, true, np.nan) * 100).mean()),
        mean_iou_matched=float(np.mean([_matched_iou(g, p) for g, p in zip(gt, preds)])),
        ap50=float(ap[:, 0].mean()), ap75=float(ap[:, 1].mean()), ap90=float(ap[:, 2].mean()),
        f1_50=float(2 * tp[:, 0].sum() / (2 * tp[:, 0].sum() + fp[:, 0].sum() + fn[:, 0].sum())),
        infer_seconds_per_image=infer_seconds / max(1, len(imgs)),
    )


def _matched_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """Mean IoU over ground-truth objects, each matched to its best-overlapping prediction."""
    pred = np.asarray(pred)
    gt_ids = np.unique(gt)[1:]
    if gt_ids.size == 0:
        return float("nan")
    ious = []
    for i in gt_ids:
        m = gt == i
        overlap = pred[m]
        overlap = overlap[overlap > 0]
        if overlap.size == 0:
            ious.append(0.0)
            continue
        j = np.bincount(overlap).argmax()
        ious.append(float((m & (pred == j)).sum()) / float((m | (pred == j)).sum()))
    return float(np.mean(ious))


# ------------------------------------------------------------------------- runner ---
def _run_case_inproc(dataset: str, model: str, force: bool = False) -> dict:
    """Train -> tune -> evaluate one (dataset, model) pair, cached to results/."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{dataset}_{model}.json"
    if path.exists() and not force:
        print(f"cached: {path.name}")
        return json.loads(path.read_text())

    session = load_session(dataset)
    print(f"{dataset} | {model} | {len(session['kept'])} eligible "
          f"-> {len(session['train'])} train / {len(session['test'])} test "
          f"(min_cells={session['min_cells']})", flush=True)

    t = train_model(session, model)
    print(f"  trained in {t['train_seconds']:.0f}s", flush=True)
    h = tune_model(session, model, t["model_path"])
    print(f"  tuned in {h['tune_seconds']:.0f}s -> {h['best_params']}", flush=True)
    e = evaluate(session, model, t["model_path"], h["best_params"])
    print(f"  R2={e['r2']:.3f}  MAE={e['mae']:.2f}  MAPE={e['mape']:.2f}%  "
          f"IoU={e['mean_iou_matched']:.3f}  AP50={e['ap50']:.3f}", flush=True)

    rec = dict(dataset=dataset, model=model, defaults=DEFAULTS[model],
               min_cells=session["min_cells"], image_shape=list(session["shape"]),
               n_eligible=len(session["kept"]), n_train=len(session["train"]),
               n_test=len(session["test"]),
               train_cells=int(sum(session["counts"][n] for n in session["train"])),
               test_cells=int(sum(session["counts"][n] for n in session["test"])),
               **{k: v for k, v in t.items() if k != "model_path"}, **h, **e)
    path.write_text(json.dumps(rec, indent=2))
    print(f"  wrote {path.name}", flush=True)
    return rec


def run_case(dataset: str, model: str, force: bool = False) -> dict:
    """Notebook entry point: run one case inside the venv that matches `model`.

    Cellpose 3 and 4 cannot share an interpreter, so training, tuning and evaluation
    all happen in a subprocess using that branch's own environment. The notebook
    kernel therefore does not need Cellpose at all.
    """
    path = OUT_DIR / f"{dataset}_{model}.json"
    if path.exists() and not force:
        print(f"cached: {path.name}  (pass force=True to recompute)")
        return json.loads(path.read_text())
    py = _worker(model)[0]
    proc = subprocess.run([py, str(Path(__file__).resolve()), dataset, model],
                          text=True, stdout=sys.stdout, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"{dataset}/{model} failed:\n" + proc.stderr[-3000:])
    return json.loads(path.read_text())


if __name__ == "__main__":
    import logging

    logging.disable(logging.INFO)
    _run_case_inproc(sys.argv[1], sys.argv[2], force=True)
