# Artist Classification with ResNet-18: Handling Long-Tail Distribution

This repository contains the solution for **Assignment #3** of the **SOI1010 Machine Learning II** course. The project focuses on classifying artworks from 50 different artists using a **Convolutional Neural Network (CNN)**, specifically addressing the challenge of severe **class imbalance (Long-Tail Distribution)**.

## 📌 Project Overview
- **Author:** Jinwook Kim (2023036299)
- **Course:** Machine Learning II (SOI1010)
- **Goal:**
  - Transition from shallow learning (SVM) to Deep Learning (CNN) for complex feature extraction.
  - Solve the **"Accuracy Paradox"** caused by imbalanced data (Van Gogh: ~500 images vs. Minority: ~19 images).
  - Improve model **generalization** and minimize the gap between Train and Validation performance.
- **Model:** Custom ResNet-18
- **Framework:** PyTorch, WandB

## 🧩 Key Challenges & Solutions

### 1. Data Imbalance (The Long-Tail Problem)
The dataset exhibits a severe disparity, with the most frequent class having ~29x more images than the least frequent.
- **Solution 1: Focal Loss**
  - Replaced Standard Cross-Entropy with **Focal Loss** ($\gamma=2.0, \alpha=0.25$) to assign higher weights to hard-to-classify examples (minority classes).
- **Solution 2: Custom Mixup Augmentation**
  - Applied **Mixup** with dynamic probabilities based on class frequency (higher mixup chance for minority classes) to encourage the model to learn intermediate features.
- **Solution 3: Multi-Crop Strategy**
  - Applied aggressive **Multi-Crop** augmentation specifically to minority classes to artificially increase their representation and diversity.

### 2. Model Selection (Why ResNet-18?)
- Given the limited dataset size (~5,300 images), deeper models like ResNet-50 posed a high risk of overfitting.
- **ResNet-18** was selected as the optimal balance between feature extraction capability and model complexity.

## 🧪 Experiment Evolution: From Overfitting to Robustness

The project followed a 4-phase experimental approach to bridge the generalization gap.

### Phase 1: Overfitting Diagnosis (Attempts #1-3)
- **Setup:** ResNet-18 + Dropout (0.5) + Early Stopping.
- **Result:** Severe overfitting. Training Accuracy reached ~90%, but Validation Accuracy stuck at ~50%.
- **Insight:** Simple regularization (Dropout) is insufficient for such small, imbalanced data.

### Phase 2: The "Specialist" Model (Attempt #4)
- **Setup:** Removed Early Stopping, trained for 100 Epochs.
- **Result:** Highest Macro F1 Score (0.454) but lowest Accuracy.
- **Insight:** The model became a "Minority Specialist," learning rare classes well but sacrificing overall accuracy.

### Phase 3: Introducing Mixup (Attempts #5-6)
- **Setup:** Applied **Mixup Augmentation**.
- **Result:** The **Generalization Gap** (Train Acc - Val Acc) drastically dropped from ~0.4 to ~0.1.
- **Insight:** The model started learning genuine features instead of memorizing images.

### Phase 4: Optimal Balance (Attempts #7-8)
- **Setup:** **Multi-Crop** (for minority) + **Custom Mixup** + ResNet-18.
- **Result:** Achieved the best balance between F1 Score (0.425) and Public Accuracy (~0.52).
- **Conclusion:** Synergy between data-centric augmentation (MultiCrop) and regularization (Mixup) yielded the most practical model.

## 📊 Results Summary

| Model ID | Method | Macro F1 | Public Score | Generalization Gap | Note |
|:---:|:---|:---:|:---:|:---:|:---|
| **#4** | Epoch 100 + Dropout | **0.454** | Low | High | Best F1 (Specialist) |
| **#6** | Custom Mixup | 0.396 | 0.475 | **Lowest (~0.1)** | Best Robustness |
| **#7** | **Multi-Crop + Mixup** | **0.425** | **0.519** | Low (~0.15) | **Final Submission** |

## 📝 Retrospective
- **Deep Learning Potential:** Observed how CNNs autonomously learn complex artistic styles that manual feature engineering (in Assignments 1 & 2) could not capture.
- **Regularization Trade-off:** Learned that excessive regularization (e.g., stacking Dropout + Mixup in Attempt #8) can lead to underfitting.
- **Metric Importance:** Confirmed that in imbalanced datasets, **Macro F1 Score** is a far more critical indicator of model health than simple Accuracy.
