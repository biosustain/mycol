### Mycol Workflow Overview

Mycol is organized into a clear, step-by-step workflow that guides you from **raw microscopy images** to **trained AI models** and **quantitative cell-level analysis**.

Below is an overview of the full process:

1. **Upload Page — Start with your data**  
   Begin by uploading your microscopy images, along with optional masks and optional AI models ([Cellpose](https://www.nature.com/articles/s41592-022-01663-4) for segmentation and [DenseNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) for classification).  
   This page establishes all inputs that downstream steps will use.

2. **Segmentation & Classification Page — Create and refine annotations**  
   View each image with its masks, generate new masks with [Cellpose](https://www.nature.com/articles/s41592-022-01663-4)/[SAM2](https://ai.meta.com/sam2/), edit any segmentation errors, and classify cells manually or with an uploaded classifier.  
   This page is the central workspace where annotated datasets are produced.

3. **Training Page — Build better models**  
   Use the annotated datasets from the Segmentation & Classification page to train new [Cellpose](https://www.nature.com/articles/s41592-022-01663-4) or [DenseNet](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf) models.  
   Newly trained models can immediately be reused to improve segmentation and classification in future sessions.

4. **Cell Analysis Page — Explore and compare cell properties**  
   Once masks and cell classes are finalized, this page lets you visualize and quantify cell-level features.  
   Generate population-level plots, compare classes, and download processed data for external analysis.

**Overall Flow:**  
**Upload → Edit + Segment + Classify → Train Models (optional) → Analyze Results**  
At any point, you can download your annotated data or trained models for reuse, publication, or integration into external workflows.

Use this overview as a roadmap to guide your analysis from raw images to actionable biological insights.
