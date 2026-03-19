import sys
import numpy as np
import torch
import os
from cellpose import models, io


import traceback


def main():
    try:
        if len(sys.argv) < 3:
            print("Usage: uv run inference_worker.py input.npz output.npz")
            sys.exit(1)

        in_path = sys.argv[1]
        out_path = sys.argv[2]

        # load data
        with np.load(in_path, allow_pickle=True) as data:
            image = data["image"]
            weights_path = str(data["weights_path"])
            channels = data["channels"].tolist()
            diameter = data["diameter"]
            if diameter == 0 or np.isnan(diameter):
                diameter = None
            else:
                diameter = float(diameter)

            cellprob_threshold = float(data["cellprob_threshold"])
            flow_threshold = float(data["flow_threshold"])
            min_size = int(data["min_size"])
            niter = int(data["niter"])

        # setup logger
        _ = io.logger_setup()

        use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
        # load model
        if os.path.exists(weights_path):
            cell_model = models.CellposeModel(
                gpu=use_gpu, pretrained_model=weights_path
            )
        else:
            cell_model = models.CellposeModel(gpu=use_gpu, model_type=weights_path)

        # eval
        sys.stdout.flush()
        masks, flows, styles = cell_model.eval(
            [image],
            channels=channels,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            niter=niter,
        )

        mask_output = masks[0]

        # save results
        np.savez_compressed(out_path, masks=mask_output)
        sys.stdout.flush()

    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()
