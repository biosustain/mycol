# **Mycol**

_A lightweight, human-in-the-loop microscopy image analysis app._

Mycol is a Streamlit-based application that makes machine-learning-assisted microscopy analysis accessible to non-specialists. It enables fast annotation, automated segmentation and classification, model fine-tuning, and quantitative phenotyping, all on a standard laptop and without coding.

---

## **Features**

### **Annotation & QC**

- Upload images and optional masks
- Manual mask drawing and editing
- SAM2-guided segmentation
- Automated Cellpose segmentation (single or batch mode)
- Interactive classification (manual or DenseNet-based)

### **Model Fine-Tuning**

- Train Cellpose (segmentation) and DenseNet (classification) models directly in the app
- Default training settings for general use
- Diagnostic outputs:
  - Loss curves
  - IoU scores
  - True vs. predicted counts
  - Accuracy, precision, F1, confusion matrix
- Download trained models and training summaries

### **Cell Metrics & Phenotyping**

- Automatic computation of cell descriptors (size, shape, elongation, compactness, etc.)
- Visual comparison of phenotypic classes
- Export plots and tabulated descriptors
- Built-in explanations for descriptor interpretation

### **Lightweight & Accessible**

- Runs locally on standard hardware
- Minimal dependencies
- Designed for small-scale workflows

---

## **Installation**

> [!WARNING] 
> `uv` is the recommended package manager for this project. It is a drop-in replacement for `pip` and `conda`.
> to install `uv`, run `pip install uv` or follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/).
> use `uv sync` to automatically create a virtual environment and install dependencies.
> use `uv run` to execute python inside the virtual environment e.g. `uv run streamlit run app.py`, or activate the environment with `source .venv/bin/activate` on unix or `.venv\Scripts\activate` on windows.

```bash
uv run streamlit run app.py
```

```bash
uv sync
```

---

## **Run the App locally**

```bash
uv run streamlit run app.py
```

---

## **Example Use Cases**

- Rapid cell counting
- Creating curated datasets of annotated images
- Automating image annotation (with human QC)
- Morphology-based phenotypic comparison

---

## **License**

MIT
