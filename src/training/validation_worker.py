import sys
import numpy as np
from pathlib import Path
from cellpose import models, metrics
import optuna
import torch
from data_split import split_train_test


def _min_size_search_bounds(masks):
    """Data-driven (lo, hi, step) for the min_size search, bounded to the small
    end of the ground-truth mask-area distribution so each step is meaningful."""
    areas = []
    for m in masks:
        vals, cnts = np.unique(np.asarray(m), return_counts=True)
        areas.append(cnts[vals != 0])  # per-instance pixel areas, background dropped
    areas = np.concatenate(areas) if areas else np.array([], dtype=int)
    if areas.size == 0:
        return 0, 500, 50
    hi = max(50, int(np.percentile(areas, 20)))  # search only the smallest ~20%
    step = max(10, hi // 10)
    return 0, hi, step


def run_optuna_tuning(images, masks, model_path, channels, n_trials):
    """Run Optuna hyperparameter optimization"""
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()

    if isinstance(model_path, str) and Path(model_path).exists():
        eval_model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
    else:
        eval_model = models.CellposeModel(gpu=use_gpu, model_type=model_path)

    # min_size range derived once from the real cell sizes in the ground-truth masks
    ms_lo, ms_hi, ms_step = _min_size_search_bounds(masks)

    results = []
    cache = {}  # param tuple -> score, so repeats skip eval

    def objective(trial):
        cellprob = round(trial.suggest_float("cellprob", -0.6, 1.0, step=0.1), 1)
        flowthresh = round(trial.suggest_float("flow_threshold", -0.2, 1.0, step=0.2), 1)
        niter = trial.suggest_categorical("niter", [0, 200, 500, 1000])
        # min_size bounded to the small end of the measured area distribution
        min_size = trial.suggest_int("min_size", ms_lo, ms_hi, step=ms_step)

        # reuse the score if this combination was already evaluated (round guards float keys)
        key = (round(cellprob, 6), round(flowthresh, 6), niter, min_size)
        if key in cache:
            print(f"Reusing cached score {cache[key]:.4f} for {key}")
            sys.stdout.flush()
            return cache[key]

        masks_pred, _, _ = eval_model.eval(
            images,
            channels=channels,
            diameter=None,
            cellprob_threshold=cellprob,
            flow_threshold=flowthresh,
            niter=niter,
            min_size=min_size,
        )
        ap = metrics.average_precision(masks, masks_pred)[0]
        score = float(np.nanmean(ap[:, 0]))

        cache[key] = score
        results.append(
            {
                "cellprob": cellprob,
                "flow_threshold": flowthresh,
                "niter": niter,
                "min_size": min_size,
                "ap_iou_0.5": score,
            }
        )
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # warm start with on-grid, in-range values matching the search space above
    study.enqueue_trial(
        {"cellprob": 0.2, "flow_threshold": 0.4, "niter": 500, "min_size": ms_lo}
    )
    study.optimize(objective, n_trials=n_trials)

    # round saved params to 1 dp (suggest_float leaves fp noise like 0.8000000000000002)
    best_params = {
        k: round(v, 1) if isinstance(v, float) else v
        for k, v in study.best_trial.params.items()
    }
    return results, best_params


def compute_validation_metrics(
    images, masks, image_names, base_model_type, tuned_model_path, channels, best_params
):
    """Compute IoU and count comparison metrics"""
    # Load models
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()

    base_model = models.CellposeModel(gpu=use_gpu, model_type=base_model_type)
    tuned_model = models.CellposeModel(gpu=use_gpu, pretrained_model=tuned_model_path)

    hp = dict(
        diameter=None,
        cellprob_threshold=float(best_params.get("cellprob", 0.2)),
        flow_threshold=float(best_params.get("flow_threshold", 0.4)),
        niter=int(best_params.get("niter", 1000)),
        min_size=int(best_params.get("min_size", 100)),
    )

    base_preds, _, _ = base_model.eval(images, channels=channels, **hp)
    tuned_preds, _, _ = tuned_model.eval(images, channels=channels, **hp)

    base_ious = []
    tuned_ious = []
    for gt, b_pred, t_pred in zip(masks, base_preds, tuned_preds):
        b_ap = metrics.average_precision([gt], [b_pred])[0]
        t_ap = metrics.average_precision([gt], [t_pred])[0]
        base_ious.append(float(np.nanmean(b_ap[:, 0])))
        tuned_ious.append(float(np.nanmean(t_ap[:, 0])))

    gt_counts = [int(np.count_nonzero(np.unique(mask))) for mask in masks]
    base_counts = [int(np.count_nonzero(np.unique(pred))) for pred in base_preds]
    tuned_counts = [int(np.count_nonzero(np.unique(pred))) for pred in tuned_preds]

    return {
        "base_ious": base_ious,
        "tuned_ious": tuned_ious,
        "gt_counts": gt_counts,
        "base_counts": base_counts,
        "tuned_counts": tuned_counts,
        "image_names": image_names,
    }


import traceback


def main():
    try:
        if len(sys.argv) != 3:
            print(
                "Usage: validation_worker.py <input.npz> <output.npz>", file=sys.stderr
            )
            sys.exit(1)

        in_path = Path(sys.argv[1])
        out_path = Path(sys.argv[2])

        with np.load(in_path, allow_pickle=True) as data:
            images_obj = data["images"]
            masks_obj = data["masks"]
            base_model = str(data["base_model"])
            tuned_model_path = str(data["tuned_model_path"])
            channels = list(data["channels"])
            do_gridsearch = bool(data["do_gridsearch"])
            n_trials = int(data.get("n_trials", 20))
            test_split = float(data["test_split"]) if "test_split" in data else 0.2
            image_names_obj = data["image_names"] if "image_names" in data else None

        images = [np.asarray(img, dtype=np.float32) for img in images_obj]  # ew
        masks = [np.asarray(mask, dtype=np.uint16) for mask in masks_obj]
        image_names = (
            [str(n) for n in image_names_obj]
            if image_names_obj is not None
            else [f"Image {i}" for i in range(len(images))]
        )

        print(f"Loaded {len(images)} images and {len(masks)} masks")
        print(f"Base model: {base_model}")
        print(f"Tuned model: {tuned_model_path}")
        print(f"Channels: {channels}")
        print(f"Do gridsearch: {do_gridsearch}")
        sys.stdout.flush()

        # Model trained on train_idx and never saw test_idx: tune on a subsample of
        # the training images, report metrics on the held-out test images.
        train_idx, test_idx = split_train_test(len(images), test_size=test_split)
        test_idx = test_idx or train_idx  # too few images to hold any out
        tune_idx = train_idx[:40]  # cap tuning at 40 training images (already shuffled)

        tune_images = [images[i] for i in tune_idx]
        tune_masks = [masks[i] for i in tune_idx]
        test_images = [images[i] for i in test_idx]
        test_masks = [masks[i] for i in test_idx]
        test_names = [image_names[i] for i in test_idx]
        print(
            f"Tuning on {len(tune_idx)} training images; "
            f"validating on {len(test_idx)} held-out images"
        )
        sys.stdout.flush()

        optuna_results = None
        best_params = {
            "cellprob": 0.2,
            "flow_threshold": 0.4,
            "niter": 1000,
            "min_size": 100,
        }

        if do_gridsearch:
            print(f"Running Optuna with {n_trials} trials...")
            sys.stdout.flush()
            optuna_results, best_params = run_optuna_tuning(
                tune_images, tune_masks, tuned_model_path, channels, n_trials
            )
            print(f"Best params: {best_params}")
            sys.stdout.flush()

        # Compute validation metrics on the held-out test images
        print("Computing validation metrics...")
        sys.stdout.flush()
        validation_metrics = compute_validation_metrics(
            test_images,
            test_masks,
            test_names,
            base_model,
            tuned_model_path,
            channels,
            best_params,
        )

        print("Validation complete!")

        # Save results
        np.savez_compressed(
            out_path,
            optuna_results=optuna_results,
            best_params=best_params,
            validation_metrics=validation_metrics,
        )

        print(f"Results saved to {out_path}")
        sys.stdout.flush()

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
