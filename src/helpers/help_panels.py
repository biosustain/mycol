from pathlib import Path
import streamlit as st

# Path to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAM_DIR = PROJECT_ROOT / "descriptor_diagrams"


def cellpose_training_plot_help():
    with st.popover("How do I interpret these graphs?"):
        st.subheader("Cellpose Training Graphs")

        st.markdown(
            """
            Below is a quick reference for the plots generated during Cellpose training.
            Use this as a guide when assessing the quality and behavior of your trained model.
            """
        )

        # --- Training vs. Test Loss ---
        with st.expander("Training vs. Test Loss"):
            st.markdown(
                """
                This plot shows how the loss changes over training epochs for both:

                - **Training loss** – how well the model fits the training images.
                - **Test (validation) loss** – how well the model generalizes to held-out test images.

                In general, you want:

                - **Both losses to decrease** as training proceeds.
                - **Test loss to plateau or reach a minimum**, then stop training around that point.

                If training loss continues to decrease while test loss starts to increase or becomes noisy,
                the model may be **overfitting**, which usually leads to worse performance on new data.
                After test loss stops improving, additional training is unlikely to help and can be harmful.
                """
            )

            st.image(
                DIAGRAM_DIR / "train_test_loss.svg",
                width='stretch',
            )

        # --- IoU Comparison ---
        with st.expander("IoU Comparison"):
            st.markdown(
                """
                **Intersection over Union (IoU)** is a standard metric for image segmentation performance.
                It measures how well the **predicted mask** overlaps with the **ground truth mask**:

                IoU = (area of overlap) / (area of union)

                In this context:

                - **Higher IoU values indicate better model performance.**
                - The bar plot shows the **mean mask IoU per image** for:
                  - the **original model** (left), and
                  - the **fine-tuned model** trained on your dataset (right).

                If the bars for the fine-tuned model are generally higher than for the original model,
                your fine-tuning has likely improved segmentation quality on your data.
                """
            )

            st.image(
                DIAGRAM_DIR / "iou.svg",
                width='stretch',
            )

        # --- Predicted vs Real Counts ---
        with st.expander("Predicted vs. Real Counts"):
            st.markdown(
                """
                These plots evaluate performance on **cell counting** tasks by comparing the **true** and
                **predicted** number of cells per image for:

                - the **original model** (left), and
                - the **fine-tuned model** (right).

                Key things to look for:

                - Points close to the **x = y line** → predictions match the true counts.
                - **High R² (coefficient of determination)** → predictions explain most of the variance in true counts.
                - **Low mean absolute error (MAE)** → on average, the predicted counts are close to the real counts.

                Models with points tightly clustered around the x = y line, **R² close to 1**, and **low MAE**
                are performing well on the counting task.
                """
            )

            st.image(
                DIAGRAM_DIR / "counts.svg",
                width='stretch',
            )


def classifier_training_plot_help():
    with st.popover("How do I interpret these graphs?"):
        st.subheader("Classifier Training Graphs")

        st.markdown(
            """
            Below is a quick reference for the plots generated during classifier training.
            Use this guide to help interpret model performance and diagnose potential issues.
            """
        )

        # --- Training vs. Test Loss ---
        with st.expander("Training vs. Test Loss"):
            st.markdown(
                """
                This plot shows how the loss changes over training epochs for both:

                - **Training loss** – how well the model fits the training data.
                - **Test (validation) loss** – how well the model generalizes to unseen data.

                Desired behavior:

                - **Both curves should decrease** as training progresses.
                - **Test loss should reach a minimum** or plateau.
                - If training loss continues to decrease while test loss increases, the model is likely **overfitting**.

                When test loss stops improving, additional training usually will not increase performance
                and may degrade generalization.
                """
            )

            st.image(
                DIAGRAM_DIR / "train_test_loss.svg",
                width='stretch',
            )

        # --- Accuracy, Precision, and F1 Scores ---
        with st.expander("Accuracy, Precision, Recall, and F1 Scores"):
            st.markdown(
                r"""
                ##### **Accuracy**
                Proportion of all predictions that are correct.  *Interpretation:* Higher accuracy means the model is correct more often overall. In multiclass settings, accuracy can look high even if some classes perform poorly.

                $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
                

                ##### **Precision**
                Of predicted positives, how many were correct. *Interpretation:* Higher precision means fewer false positives. In multiclass results (macro/weighted), it reflects how reliably the model’s positive predictions are correct across all classes.  
                
                $$\text{Precision} = \frac{TP}{TP + FP}$$
                

                ##### **Recall**
                Of actual positives, how many were correctly identified. *Interpretation:* Higher recall means fewer missed cases. As an overall metric, it summarizes how well the model captures true instances across all classes.
                
                $$\text{Recall} = \frac{TP}{TP + FN}$$
                

                ##### **F1 Score**
                Harmonic mean of precision and recall. *Interpretation:* Higher F1 means the model maintains a good balance between precision and recall. Useful in multiclass settings because it is less affected by class imbalance than accuracy alone.  
                
                $$F1 = 2 \cdot \frac{P \cdot R}{P + R}$$
                """
            )

            st.image(
                DIAGRAM_DIR / "acc_prec_f1.svg",
                width='stretch',
            )

        # --- Confusion Matrix ---
        with st.expander("Confusion Matrix"):
            st.markdown(
                """
                A confusion matrix shows **how often each class is predicted as each other class**.

                - **Rows** represent the *true* classes.
                - **Columns** represent the *predicted* classes.
                - Perfect classification would produce strong diagonal values and zeros elsewhere.

                How to interpret:
                - **High diagonal values** → correct predictions.
                - **Off-diagonal values** → misclassifications (which classes get confused).
                - Blocks of confusion may indicate:
                    - insufficient training data for certain classes,
                    - classes with very similar cells,
                    - or inadequate model capacity.

                Use this plot to identify which classes require more attention or additional data augmentation.
                """
            )

            st.image(
                DIAGRAM_DIR / "confusion_matrix.svg",
                width='stretch',
            )


def shape_metric_help():
    st.subheader("Shape Descriptor Reference")

    st.markdown(
        """
        Below is a quick reference for the shape descriptors computed from each labeled region.
        Use this as a guide when interpreting the measurements for your segmented cells.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Notation:**  
            - \(A\): area (number of pixels in the object)  
            - \(P\): perimeter (length of the object's boundary)  
            - \(a, b\): semi-major and semi-minor axes of the best-fit ellipse  

            All quantities are reported in pixel units.
            """
        )

    with col2:
        st.image(
            DIAGRAM_DIR / "plot_download.svg",
            width='stretch',
        )

    # --- Textual definitions -------------------------------------------------
    metrics = [
        {
            "Name": "area (A)",
            "What it describes": "Size of the object in pixels.",
            "How it is calculated": "Number of pixels inside the masked region.",
        },
        {
            "Name": "perimeter (P)",
            "What it describes": "Length of the object's boundary.",
            "How it is calculated": "Length of the outer contour of the masked region.",
        },
        {
            "Name": "major / minor axis lengths",
            "What it describes": (
                "The longest (major) and shortest (minor) axes of the best-fit ellipse. "
                "Larger major axis values (relative to object size) indicate a more elongated object."
            ),
            "How it is calculated": (
                "Lengths of the major and minor axes of the ellipse with the same second moments "
                "as the region. These correspond to major_axis_length and minor_axis_length."
            ),
        },
        {
            "Name": "circularity / compactness / roundness",
            "What it describes": (
                "Three related measures of how circle-like and compact a shape is. "
                "Circularity and roundness are high (≈ 1) for circle-like objects and decrease with "
                "elongation or irregular boundaries. Compactness is the inverse of circularity and "
                "increases with boundary irregularity."
            ),
            "How it is calculated": (
                "\n- Circularity: 4 · π · A / P²\n"
                "- Compactness: P² / (4 · π · A)\n"
                "- Roundness: 4 · A / (π · major_axis_length²)"
            ),
        },
        {
            "Name": "aspect ratio / elongation / eccentricity",
            "What it describes": (
                "Three related measures of how stretched the best-fit ellipse is. "
                "Aspect ratio compares the major and minor axis lengths (≥ 1). "
                "Elongation normalizes the difference between axes into the range 0 to 1. "
                "Eccentricity measures how far the ellipse deviates from a circle, also ranging from 0 to 1."
            ),
            "How it is calculated": (
                "\n- Aspect ratio = major_axis_length / minor_axis_length\n"
                "- Elongation = (major_axis_length − minor_axis_length)\n"
                "               / (major_axis_length + minor_axis_length)\n"
                "- Eccentricity = √(1 − (b² / a²)), using semi-axes a (major) and b (minor)"
            ),
        },
        {
            "Name": "solidity",
            "What it describes": (
                "How filled the object is relative to its convex hull. "
                "A value of 1 indicates a perfectly convex shape; lower values indicate concavities or "
                "irregular boundaries."
            ),
            "How it is calculated": "area / convex_area",
        },
        {
            "Name": "extent",
            "What it describes": (
                "Fraction of the bounding box area occupied by the object. "
                "Values near 1 indicate that the object nearly fills its bounding box."
            ),
            "How it is calculated": "area / bounding_box_area",
        },
    ]

    # --- Per-metric expanders ("popovers") -----------------------------------

    for m in metrics:
        with st.expander(m["Name"]):
            st.markdown(
                f"**What it describes:** {m['What it describes']}\n\n"
                f"**How it is calculated:** {m['How it is calculated']}"
            )

            if m["Name"] == "circularity / compactness / roundness":
                st.image(
                    DIAGRAM_DIR / "circularity_compactness_roundness.svg",
                    width='stretch',
                )

                st.caption(
                    "For the same area A, shapes with longer perimeters P have lower circularity "
                    "and roundness, and higher compactness. All three metrics summarize how "
                    "circle-like and compact the boundary is."
                )

            if m["Name"] == "major / minor axis lengths":
                st.image(DIAGRAM_DIR / "axes.svg", width='stretch')

                st.caption(
                    "The major axis is the longest diameter of the best-fit ellipse; "
                    "the minor axis is the shortest diameter perpendicular to it. "
                    "Their lengths are reported as major_axis_length and minor_axis_length."
                )

            if m["Name"] == "aspect ratio / elongation / eccentricity":
                st.image(
                    DIAGRAM_DIR / "elongation_eccentricity_aspect_ratio.svg",
                    width='stretch',
                )

                st.caption(
                    "All three metrics describe how stretched the best-fit ellipse is. "
                    "As the major axis increases relative to the minor axis, aspect ratio, "
                    "elongation, and eccentricity all increase."
                )

            if m["Name"] == "solidity":
                st.image(DIAGRAM_DIR / "solidity.svg", width='stretch')

                st.caption(
                    "Solidity = area / convex_area. A convex shape matches its convex hull (solidity ≈ 1). "
                    "Indentations or holes reduce the area relative to the hull, lowering solidity."
                )

            if m["Name"] in ["extent"]:
                st.image(DIAGRAM_DIR / "extent.svg", width='stretch')

                st.caption(
                    "Extent measures how much of the bounding box area the object occupies. "
                    "The bounding-box aspect ratio compares its longer side to its shorter side; "
                    "thin, elongated boxes have large aspect ratios."
                )
