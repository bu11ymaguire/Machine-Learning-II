# MNIST k-NN Classifier with PyTorch

This repository contains the solution for **Assignment #1** of the **SOI1010 Machine Learning II** course. The project focuses on implementing and optimizing the k-Nearest Neighbors (k-NN) algorithm to classify MNIST handwritten digits using PyTorch.

## 📌 Project Overview
- **Author:** Jinwook Kim (2023036299)
- **Course:** Machine Learning II (SOI1010)
- **Goal:** Implement k-NN from scratch, optimize it using PyTorch's tensor operations, and perform hyperparameter tuning.
- **Tools:** Python, PyTorch, Google Colab

## 🚀 Key Implementations

The project explores the evolution of k-NN implementation through the following steps:

### 1. Iterative Implementation (Naive Approach)
- Implemented k-NN ($k=5$) using standard Python loops.
- Manually calculated Euclidean distance (L2) for a single test image against all training images.
- **Limitation:** Computationally expensive and slow due to lack of vectorization.

### 2. Broadcasting Implementation (Single Image)
- Utilized PyTorch's **Broadcasting** mechanism to calculate distances.
- Expanded dimensions using `unsqueeze(0)` to enable vectorized operations between a test image and the training set.
- Replaced manual sorting with `torch.topk` for efficiency.

### 3. Memory Optimization (Batch Processing)
- **Issue:** Attempting to broadcast all validation images ($6000$) against all training images ($54000$) directly resulted in an **Out-Of-Memory (OOM)** error due to the intermediate tensor shape `[6000, 54000, 28, 28]`.
- **Solution:** - Flattened images to 1D vectors (`28x28` -> `784`).
  - Used **`torch.cdist`** to efficiently compute the pairwise Euclidean distance matrix.
  - Determined final labels using **`torch.mode`** (majority vote).
  - This approach successfully processed the entire dataset without memory overflow.

## 📊 Experiment & Results

### Hyperparameter Tuning
- **Parameters Tuned:** - `val_ratio` (Validation Set Ratio): 0.01 ~ 0.30
  - `k` (Number of Neighbors): 5 ~ 20
- **metric:** Accuracy & F1-Score

### Findings
- **Best Performance:** `val_ratio=0.01`, `k=7` (Accuracy: ~97.8%)
- **Stability:** `k=5` showed the most consistent performance across various validation ratios.
- **Trade-off:** While `k=7` had the highest peak accuracy, `k=5` is considered more robust against overfitting to specific data splits.

### Final Test Results
Evaluated on the test set using the full training data:
| Model | Accuracy | F1-Score | Note |
|-------|----------|----------|------|
| **k=7** | **96.94%** | **0.9694** | Best Peak Performance |
| **k=5** | 96.88% | 0.9687 | Most Stable |

## 🧐 Error Analysis
An analysis of misclassified digits (specifically for `k=7`) revealed:
- **Digit '2':** Generally strong, but suffers from occasional False Negatives.
- **Digit '7' vs '9':** High confusion rates due to handwriting similarities (e.g., rounded '7' looking like '9').
- **Digit '8':** Higher False Negatives compared to other digits.

## 💻 Usage
The implementation is available in the Jupyter Notebook/Google Colab format.

**Colab Link:** [View Notebook](https://colab.research.google.com/drive/1ZXRRvVvv7evmu_sBHgf1eiTMk8XB_d1P?usp=sharing)

## 📝 Conclusion
This assignment highlighted the importance of using optimized library functions (`torch.cdist`, `torch.topk`) over manual implementation or naive broadcasting to handle large-scale tensor operations efficiently in terms of both speed and memory.
