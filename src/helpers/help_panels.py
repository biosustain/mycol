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
                width="stretch",
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
                width="stretch",
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
                width="stretch",
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
                width="stretch",
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
                width="stretch",
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
                width="stretch",
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

            All quantities are reported in pixels unless a conversion factor is applied.
            """
        )

    with col2:
        st.image(
            DIAGRAM_DIR / "plot_download.svg",
            width="stretch",
        )

    # --- Textual definitions -------------------------------------------------
    metrics = [
        {
            "Name": "area (A)",
            "What it describes": "Size of the object in pixels.",
        },
        {
            "Name": "perimeter (P)",
            "What it describes": "Length of the object's boundary.",
        },
        {
            "Name": "major / minor axis lengths",
            "What it describes": (
                "The longest (major) and shortest (minor) axes of the best-fit ellipse. "
                "Larger major axis values (relative to object size) indicate a more elongated object."
            ),
        },
        {
            "Name": "feret diameter",
            "What it describes": (
                "The largest distance between any two points on the object's boundary, "
                "as if measured with callipers. Unlike the major axis length it assumes "
                "no particular shape, so it is the more reliable size measure for "
                "irregular or bent objects."
            ),
        },
        {
            "Name": "circularity / roundness",
            "What it describes": (
                "Two related measures of how circle-like and compact a shape is. "
                "Both are high (≈ 1) for circle-like objects and decrease with "
                "elongation or irregular boundaries."
            ),
        },
        {
            "Name": "eccentricity",
            "What it describes": (
                "How stretched the best-fit ellipse is, from 0 to 1. A value of 0 is a "
                "perfect circle; values approaching 1 are increasingly elongated. It "
                "compares the two axes, so it does not change with the size of the cell."
            ),
        },
        {
            "Name": "solidity",
            "What it describes": (
                "How filled the object is relative to its convex hull. "
                "A value of 1 indicates a perfectly convex shape; lower values indicate concavities or "
                "irregular boundaries."
            ),
        },
        {
            "Name": "extent",
            "What it describes": (
                "Fraction of the bounding box area occupied by the object. "
                "Values near 1 indicate that the object nearly fills its bounding box."
            ),
        },
    ]

    # --- Per-metric expanders ("popovers") -----------------------------------

    for m in metrics:
        with st.expander(m["Name"]):
            st.markdown(f"**What it describes:** {m['What it describes']}")

            if m["Name"] == "circularity / roundness":
                st.image(
                    DIAGRAM_DIR / "circularity_roundness.svg",
                    width="stretch",
                )

                st.caption(
                    "For the same area A, shapes with longer perimeters P have lower circularity "
                    "and roundness. Both metrics summarize how "
                    "circle-like and compact the boundary is."
                )

            if m["Name"] == "major / minor axis lengths":
                st.image(DIAGRAM_DIR / "axes.svg", width="stretch")

            if m["Name"] == "feret diameter":
                st.image(DIAGRAM_DIR / "feret_diameter.svg", width="stretch")

                st.caption(
                    "A calliper reading depends on the direction it is taken in: Fh "
                    "across and Fv down. Turning the same object changes both. Mycol "
                    "reports the maximum Feret diameter, the largest reading over every "
                    "direction, so it does not depend on how the cell happens to lie."
                )

                st.caption(
                    "The major axis is the longest diameter of the best-fit ellipse; "
                    "the minor axis is the shortest diameter perpendicular to it. "
                    "Their lengths are reported as major_axis_length and minor_axis_length."
                )

            if m["Name"] == "eccentricity":
                st.image(
                    DIAGRAM_DIR / "eccentricity.svg",
                    width="stretch",
                )

                st.caption(
                    "Eccentricity describes how stretched the best-fit ellipse is. "
                    "As the major axis grows relative to the minor axis, eccentricity "
                    "rises from 0 towards 1."
                )

            if m["Name"] == "solidity":
                st.image(DIAGRAM_DIR / "solidity.svg", width="stretch")

                st.caption(
                    "Solidity = area / convex_area. A convex shape matches its convex hull (solidity ≈ 1). "
                    "Indentations or holes reduce the area relative to the hull, lowering solidity."
                )

            if m["Name"] in ["extent"]:
                st.image(DIAGRAM_DIR / "extent.svg", width="stretch")

                st.caption(
                    "Extent measures how much of the bounding box area the object occupies. "
                    "The bounding-box aspect ratio compares its longer side to its shorter side; "
                    "thin, elongated boxes have large aspect ratios."
                )
