This repository contains the code used for the paper: **Principal Components for Neural Network Initialization: A Novel Approach to Explainability and Efficiency**.

The codebase is organized around evaluating the **PCSINIT (Principal Component-based Initialization)** method across different aspects, including performance, interpretability, stability, and geometric analysis.

***

## Repository Contents

The main functionalities and experiments are implemented in the following Jupyter Notebooks:

### **Training and Performance Evaluation**

* **`psinit_nb.ipynb`**: Contains the core logic for running neural network experiments, primarily focusing on comparing models initialized with different classical methods (`'he'`, `'xavier'`, `'orthogonal'`) and demonstrating the architecture's use on tabular datasets (e.g., the 'heart' dataset). This notebook includes utilities for:
    * Running multiple trials (`nruns = 10`) for statistical robustness.
    * Calculating and plotting mean and confidence intervals (CI) for training metrics.

### **Interpretability and Feature Importance (SHAP/AOPC): in the xAI folder**

* **AOPC Calculation (`Aopc_NN.ipynb`, `Aopc_PCsINIT.ipynb`)**: These notebooks measure **Area Over Perturbation Curve (AOPC)**, an interpretability metric, using **SHAP (SHapley Additive exPlanations)** values.
    * The tests evaluate the change in performance (AOPC) when perturbing the top-$k$ most important features (for $k=1$ to $5$).
    * Different perturbation methods are tested, including `'mean'`, `'median'`, `'zero'`, and `'marginal'`.
* **SHAP Explanations (`shapley_NN.ipynb`, `shapley_PCsInit_PCANN.ipynb`)**: Generate **SHAP summary plots** to visualize global and local feature importance.
    * These notebooks compare the standard HE-initialized network (`shapley_NN.ipynb`) with the **PCSINIT** (PCA-initialized network) and a network trained on dimensionality-reduced **PCA-transformed features** (`PCAFFW`).
* **Synthetic Data Feature Recovery (`NN_synthetic_data.ipynb`, `synthetic_data_PCsInit_NN.ipynb`)**: These modules are used to test how accurately SHAP can recover the **known ground-truth important features** in artificially generated datasets, offering a validation of the explanation quality.

### **Stability and Geometric Analysis**

* **Stability Analysis (`Stability_PCsInit_NN_.ipynb`)**: Calculates the **explanation variance ($\sigma^2_{exp}$)**, a measure of how stable the SHAP explanations are across different training runs for the same model configuration.
* **Principal Angles Analysis (`principal_angles.ipynb`)**: Computes the **principal angles** and **cosine similarity** between the initial PCA-derived weights (`W_r`) and the final optimized weights (`W_optimal`) of the first hidden layer. This provides a geometric measure of how much the learned input subspace drifts during training.
