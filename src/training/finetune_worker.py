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

            # debug prints
            print(
                f"DEBUG: loaded images_in type={type(images_in)} shape={getattr(images_in, 'shape', 'N/A')} dtype={getattr(images_in, 'dtype', 'N/A')}"
            )
            print(
                f"DEBUG: loaded masks_in type={type(masks_in)} shape={getattr(masks_in, 'shape', 'N/A')} dtype={getattr(masks_in, 'dtype', 'N/A')}"
            )

            # Ensure we have a list of numpy arrays with numeric dtypes
            if isinstance(images_in, np.ndarray) and images_in.dtype == object:
                images = [np.asanyarray(im).astype(np.float32) for im in images_in]
            elif isinstance(images_in, np.ndarray):
                images = [im.astype(np.float32) for im in images_in]
            else:
                images = [np.asanyarray(im).astype(np.float32) for im in images_in]

            if isinstance(masks_in, np.ndarray) and masks_in.dtype == object:
                masks = [np.asanyarray(ma).astype(np.uint16) for ma in masks_in]
            elif isinstance(masks_in, np.ndarray):
                masks = [ma.astype(np.uint16) for ma in masks_in]
            else:
                masks = [np.asanyarray(ma).astype(np.uint16) for ma in masks_in]

            print(
                f"DEBUG: processed images len={len(images)} first_type={type(images[0])} first_shape={images[0].shape} first_dtype={images[0].dtype}"
            )
            print(
                f"DEBUG: processed masks len={len(masks)} first_type={type(masks[0])} first_shape={masks[0].shape} first_dtype={masks[0].dtype}"
            )

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
