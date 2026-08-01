# Cellpose 3 vs Cellpose 4

Does mycol ship Cellpose 3 or Cellpose 4? These notebooks train both, at each generation's own mycol
defaults, on four annotated datasets, and compare accuracy against training cost.

## Layout

| notebook | dataset | model |
|---|---|---|
| `01_cs1_cp3.ipynb` | cs1 | Cellpose 3 |
| `02_cs1_cp4.ipynb` | cs1 | Cellpose 4 (Cellpose-SAM) |
| `03_bright_ecoli_cp3.ipynb` | bright_ecoli | Cellpose 3 |
| `04_bright_ecoli_cp4.ipynb` | bright_ecoli | Cellpose 4 (Cellpose-SAM) |
| `05_s_aureus_cp3.ipynb` | s_aureus | Cellpose 3 |
| `06_s_aureus_cp4.ipynb` | s_aureus | Cellpose 4 (Cellpose-SAM) |
| `07_mcount_cp3.ipynb` | mcount | Cellpose 3 |
| `08_mcount_cp4.ipynb` | mcount | Cellpose 4 (Cellpose-SAM) |
| `09_comparison.ipynb` | all | side-by-side plots |

`common.py` holds the shared harness. `results/` holds one JSON per case; delete a file to recompute
it. Notebooks 01–08 are independent and can run in any order; 09 reads their output.

## Protocol

Both models are trained and tuned by **mycol's own worker scripts** (`unified_worker.py`), driven
through the same npz interface the app uses — not a re-implementation. Cellpose 3's workers live on
the `main` branch and Cellpose 4's on `cp4`, so each runs from its own checkout with its own
environment:

| | Cellpose 3 | Cellpose 4 |
|---|---|---|
| checkout | `../mycol-main-cp3` (worktree of `main`) | this repo (`cp4`) |
| cellpose | 3.1.1 | 4.2.1.1 |
| base model | `cyto2` | `cpsam_v2` |
| epochs | 100 | 100 |
| learning rate / weight decay | 0.1 / 1e-4 | 1e-5 / 0.1 |
| batch size | 8 | 1 |
| images per epoch | all training images | 8 |
| Optuna trials | 20 | 20 |

Every value above is mycol's own default for that generation, taken from each branch's
`fine_tune_panel.py`. Each branch's Optuna search space is used unchanged — including Cellpose 4's
`diameter` search, which Cellpose 3 does not have.

**The train/test split is the saved session's**, rebuilt from `image_metadata.json` key order (mycol's
`ordered_keys()`, which is upload order and not always alphabetical) with the session's own
`min_cells_per_image`. That parameter decides which images are eligible, so it has to come from the
session for the split to match; everything else uses the defaults above.

Because Cellpose 3 and 4 cannot share an interpreter, the whole pipeline runs in a subprocess using
the matching environment. **This notebook therefore runs under any Python kernel** — it needs no
Cellpose itself.

## Metrics

On each dataset's held-out test images: **R²**, **MAE** and **MAPE** on per-image cell counts;
**mean IoU** of matched objects; **AP@0.5 / 0.75 / 0.9**; **F1@0.5**. Training and tuning wall-clock
are recorded separately, along with inference seconds per image.

Note that mycol's own validation worker reports AP@0.5 under the name `base_ious`. The IoU column
here is a genuine mean IoU over matched objects, computed separately, so the two are not the same
quantity.
