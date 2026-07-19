import sys
import numpy as np
import torch
from cellpose import models, train, io
from sklearn.model_selection import train_test_split


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
            nimg_per_epoch = int(data["nimg_per_epoch"])
            channels = data["channels"].tolist()
            min_train_masks = (
                int(data["min_train_masks"]) if "min_train_masks" in data else 5
            )

        # split data
        train_images, test_images, train_masks, test_masks = train_test_split(
            images, masks, test_size=0.2, random_state=42, shuffle=True
        )

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
