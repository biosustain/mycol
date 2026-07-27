import sys
import numpy as np
import torch
from cellpose import models, train, io
from data_split import split_train_test


import traceback


def main():
    try:
        if len(sys.argv) < 3:
            print("Usage: uv run finetune_worker.py input.npz output.npz")
            sys.exit(1)

        in_path = sys.argv[1]
        out_path = sys.argv[2]

        # load data
        with np.load(in_path, allow_pickle=True) as data:
            images_in = data["images"]
            masks_in = data["masks"]

            # Ensure we have a list of numpy arrays with numeric dtypes
            images = [np.asanyarray(im).astype(np.float32) for im in images_in]
            masks = [np.asanyarray(ma).astype(np.uint16) for ma in masks_in]

            base_model = str(data["base_model"])
            epochs = int(data["epochs"])
            learning_rate = float(data["learning_rate"])
            weight_decay = float(data["weight_decay"])
            batch_size = int(data["batch_size"]) if "batch_size" in data else 8
            nimg_per_epoch = int(data["nimg_per_epoch"]) or None  # 0 = all images
            channels = data["channels"].tolist()
            min_train_masks = (
                int(data["min_train_masks"]) if "min_train_masks" in data else 5
            )

        # shared split; the model never trains on the test images
        train_idx, test_idx = split_train_test(len(images))
        train_images = [images[i] for i in train_idx]
        train_masks = [masks[i] for i in train_idx]
        test_images = [images[i] for i in test_idx] or None  # [] -> None (no test set)
        test_masks = [masks[i] for i in test_idx] or None

        # Setup logger
        _ = io.logger_setup()

        # Load model
        init_model = None if base_model == "scratch" else base_model
        # Note: Use Cellpose 3 logic here (which is what we expect in this environment)

        use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        cell_model = models.CellposeModel(gpu=use_gpu, model_type=init_model)

        model_name = f"{base_model}_finetuned.pt"

        # Train
        # train_seg returns: filename, train_losses, test_losses
        sys.stdout.flush()
        new_path, train_losses, test_losses = train.train_seg(
            cell_model.net,
            train_data=train_images,
            train_labels=train_masks,
            test_data=test_images,
            test_labels=test_masks,
            channels=channels,
            n_epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            SGD=True,
            batch_size=batch_size,
            nimg_per_epoch=nimg_per_epoch,
            model_name=model_name,
            save_path=None,
            min_train_masks=min_train_masks,
        )

        # Save results
        state_dict = cell_model.net.state_dict()

        np.savez_compressed(
            out_path,
            state_dict=state_dict,
            train_losses=train_losses,
            test_losses=test_losses,
            model_name=model_name,
        )
        sys.stdout.flush()

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
