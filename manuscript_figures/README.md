# Manuscript figures

One directory per figure, one notebook per figure. Each notebook is self-contained: it draws every
panel of its figure, assembles them, and writes the result, without importing anything from another
figure's folder or from a shared helper module. `assets/` holds the inputs nothing can generate, and
`analyses/` the working that produced those inputs.

```
manuscript_figures/
├── manuscript/
│   ├── figure1/   figure1.ipynb   assets/
│   ├── figure2/   figure2.ipynb   assets/  analyses/     ← case_study_1/
│   ├── figure3/   figure3.ipynb   assets/  analyses/     ← case_study_2/
│   └── figure4/   figure4.ipynb   assets/  analyses/     ← case_study_3/
└── SI/
    ├── figureS1/ … figureS7/, figureS9/   screenshot + notebook
    ├── figureS8/  figureS8.ipynb  assets/  analyses/
    └── figureS10/ figureS10.ipynb assets/
```

Each notebook writes into its own `output/` and ends by listing what it produced and showing the
figure.

## Running

Use the `mycol_colonies_env` kernel, and run a notebook from inside its own directory - every path is
relative to it:

```bash
cd manuscript_figures/manuscript/figure2
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=mycol_colonies_env figure2.ipynb
```

Figures 2-4 take about twenty seconds each and need their case study on disk; everything else takes
seconds and needs only its own `assets/`.

**The notebooks are stored with their outputs stripped**, so they open empty - run one to see its
figure, or look in that figure's `output/`. Strip them again before committing:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to notebook --inplace <notebook>
```

## What each notebook does

| Figure | Panels it draws | Needs |
|---|---|---|
| 1 | none - copies the Inkscape master from `assets/` | nothing |
| 2 | **a** workflow strip, **b-d** plate photos, **e-g** true vs predicted CFU count | all three `case_study_1/` sessions |
| 3 | **a** workflow strip, **b-f** spore field, counts, confusion matrix, timing | `case_study_2/` session |
| 4 | **a** workflow strip, **b-f** larvae crops, confusion matrix, length, area, t-SNE | `case_study_3/` session |
| S1-S7, S9 | none - copies a screenshot from `assets/` | nothing |
| S8 | rater-pair mean absolute difference | nothing |
| S10 | manual vs Mycol length, % difference | nothing |

Figures 2-4 run in four stages inside one notebook: the workflow strip (panel **a**, drawn from
seeded geometry), the data panels, assembly onto a 1788 px canvas with the panel letters, then a 300
dpi rasterisation of the finished `.svg` by the headless Chrome kaleido installs. The raster is what
the manuscript embeds.

## Drawn vs copied

Everything is drawn by its notebook except three things, which nothing here can produce:

| File | What it is |
|---|---|
| `figure1/assets/Figure_1.svg` / `.png` | artwork drawn in Inkscape, exported at 500 dpi |
| `figureS1-S7`, `figureS9` `assets/*.png` | screenshots of the running app |
| `figure2/assets/*_example_image.jpg` | plate photographs for panels **b** and **d** |

> **`assets/` is committed, precisely because it cannot be regenerated.** The eight SI screenshots
> and Figure 1's Inkscape master exist nowhere else. Only each figure's `output/` is gitignored -
> rerun the notebook to get it back.

## Duplicated inputs

Two files are used by two figures each and are stored with both, rather than shared:

* `20250908_All_data_manual_vs_tool.xlsx` — Figure 3 panel f and Supplementary Figure 8
* `manual_vs_mycol_larvae_measurements.csv` — Figure 4 panel d and Supplementary Figure 10

They are 100 KB and 3 KB. Keeping a copy in each directory is the price of a figure never depending
on another figure's folder; if one is ever re-derived, update both.

## `analyses/`

Working that is not on the figure path but produced an input, or a number the manuscript states.
Nothing in the figure build imports them.

| Where | Notebook | What it gives |
|---|---|---|
| `figure2/analyses/` | `cellpose_mcount_evaluation.ipynb` | **the mCount comparison** - per whole well image, `-1` wells excluded, fine-tuning wells removed. On the 494 held-out wells panel g plots: **Cellpose 1.92% vs MCount 3.74% vs NICE 17.31%**. Inference runs the app pipeline, i.e. the 512x512 upload resize the model was fine-tuned at. Writes `assets/cellpose_count_evaluation.csv` |
| `figure3/analyses/` | `cs2_cellpose3_evaluation.ipynb` | `assets/cs2_cellpose3_results.csv`, the held-out counts panel d plots |
| `figure4/analyses/` | `morphology_metrics_visualisation.ipynb` | the PCA/UMAP/t-SNE exploration panel f's settings came from |
| `figureS8/analyses/` | `mycol_vs_manual.ipynb` | ICC between raters, per-image variance, and the timing behind Figure 3f |

Their scratch output goes to `analyses/output/`.

**Cellpose 3 vs Cellpose 4** is a separate question with its own directory: `cp3_vs_cp4/` on the
`cp4` branch. It is kept apart because the two versions cannot share an environment - Cellpose 4
dropped the `channels` argument and cannot load a Cellpose 3 `.pt`.

## Things to know

1. **Figure 1 cannot be rebuilt from code** - it is hand-drawn artwork. Edit `assets/Figure_1.svg`.
2. **Figure 1 reads "different celltypes"** where it should be "cell types".
3. **Supplementary Figure 6b** is labelled "Mean IoU" on the axis but "AP0.5" in the caption. The
   session's `cellpose_iou_comparison.json` titles its y axis `Mean IoU`.
4. **Supplementary Figures S6, S7 and S9 could be generated** - every panel in them is a Plotly
   figure the app stores in the saved session. Each notebook lists which.
5. **Chrome clips a headless page by ~87 CSS px** of reserved window furniture, so the rasteriser
   renders into a window 200 px taller than the figure and crops back. Without that the bottom row
   of panels loses its axis labels.
6. **The panel cells force the Agg backend.** Left to itself matplotlib picks the macOS backend on a
   Retina machine, applies 2x device scaling and snaps each figure's size to whole device pixels,
   which shifts a panel by a couple of pixels.
7. **Figure 2 has no E. coli plate photograph.** Panel **c** uses `ecoli_46.tif` from the E. coli
   session, which is why that figure needs `case_study_1/` for its pictures as well as its numbers.
8. **The Supplementary Figure 10 in the submitted SI document is the older Plotly version.** The
   image this notebook draws needs dropping into the document.
9. **Figure 4's bottom row is pinned to a recorded plot rectangle** (`COMMITTED_RECT`), because the
   measurement depends on the renderer. Pass `rect=None` to `save_bottom_row()` to re-measure it.
