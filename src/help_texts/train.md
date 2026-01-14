### 3. Train your own analysis models

This page allows you to train new machine learning models using the datasets you’ve uploaded or created on the **Segmentation & Classification** page.

You can choose to train either:

- **A Cellpose segmentation model** – to improve or customize how cells are automatically detected and outlined in your images.  
- **A DenseNet classification model** – to fine-tune how cells are categorized and labeled based on their features.

By default, Mycol provides sensible training parameters suitable for most datasets.  
However, you can adjust these parameters and run **hyperparameter optimization** to explore how different settings affect model performance.

After training completes, the app displays a set of **performance plots** that show:

- How the training progressed over time  
- How well the model fits and generalizes to your data  
- Metrics such as accuracy, loss, and validation performance

Once trained, you can:

- **Use the new model immediately** on the **Segmentation & Classification** page to analyze images automatically  
- **Download the trained model**, along with the training data and parameters, for reuse in publications or for future analyses

This page gives you a simple and reproducible way to build, evaluate, and deploy custom AI models tailored to your specific image data.
