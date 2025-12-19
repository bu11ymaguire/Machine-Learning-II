# CIFAR-10 Classification: SVM & Softmax Regression

This repository contains the solution for **Assignment #2** of the **SOI1010 Machine Learning II** course. The project involves implementing **Soft-Margin SVM** and **Multinomial Logistic Regression (Softmax)** from scratch using PyTorch to classify images from the **CIFAR-10** dataset.

## 📌 Project Overview
- **Author:** Jinwook Kim (2023036299)
- **Course:** Machine Learning II (SOI1010)
- **Goal:**
  - Implement Binary & Multiclass Classifiers (SVM, Softmax) without using high-level library functions for the core logic.
  - Analyze the impact of initialization, normalization, and class imbalance on model performance.
- **Dataset:** CIFAR-10 (10 classes, RGB images)

## 🚀 Key Implementations & Troubleshooting

### 1. Binary Classification (Airplane vs. Ship) via Soft-Margin SVM
**Objective:** Classify two visually similar classes (Airplane vs. Ship) using a custom SVM with Hinge Loss.

- **Initial Approach (Failed): Geometric Initialization**
  - *Idea:* Initialize weights ($w$) based on the vector difference between two randomly selected samples from each class.
  - *Result:* **Severe Overfitting.** The model was biased toward those specific samples and failed to generalize.
  - *Fix:* Switched to **Random Initialization** and applied **Data Normalization** (Standardization).
  - *Outcome:* Accuracy improved significantly (Best Test Acc: **72.20%**).

### 2. Multiclass Classification via One-vs-All SVM
**Objective:** Extend the binary SVM to 10 classes using the **One-vs-All (OvA)** strategy.

- **Observation: The Accuracy Paradox**
  - Validation accuracy reached **~90%**, but Test accuracy was significantly lower.
  - *Analysis:* In a 1-vs-9 setting, the data is heavily imbalanced (Positive: 10%, Negative: 90%). The model learned to simply predict "Not Target" (Negative Class) to achieve high accuracy without actually learning the features of the target class.
  - *Insight:* High accuracy in OvA does not guarantee good generalization in multiclass tasks due to label imbalance.

### 3. Multinomial Logistic Regression (Softmax) from Scratch
**Objective:** Implement Softmax Classifier and Cross-Entropy Loss manually.

- **Numerical Stability Implementation:**
  - Implemented the **Log-Sum-Exp** trick ($x - \max(x)$) to prevent numerical overflow/underflow when calculating exponentials.
  - Combined Softmax and Cross-Entropy into a single function `L_SCE` for stability.
- **Comparison with PyTorch:**
  - The scratch implementation achieved **40.49%** accuracy, which is comparable to the PyTorch built-in `nn.CrossEntropyLoss` model (**40.85%**).
  - Validated that the manual implementation is mathematically correct and optimized.

## 📊 Experiment Results

| Method | Task | Best Accuracy | Key Note |
|--------|------|---------------|----------|
| **Binary SVM** | Airplane vs. Ship | **72.20%** | Demonstrated importance of Normalization |
| **OvA SVM** | 10-Class | ~90% (Val) / Low (Test) | Suffered from Class Imbalance issue |
| **Softmax (Scratch)** | 10-Class | **40.49%** | Robust implementation with LogSumExp |
| **Softmax (PyTorch)** | 10-Class | **40.85%** | Benchmark comparison |

## 💻 Usage
The full implementation, including data visualization and hyperparameter tuning logs, is available in the notebook.

**Colab Link:** [View Notebook](https://colab.research.google.com/drive/1JB4B1ezh5W695VJQyKI01hb_NM3kR1up?usp=sharing)

## 📝 Conclusion
- **Initialization Matters:** Biased initialization (based on single samples) can ruin model performance. Standard random initialization is often safer.
- **Metric Trap:** In One-vs-All classification, accuracy is a misleading metric due to class imbalance.
- **Linear Limitations:** Linear models (SVM, Softmax) on raw pixels have inherent limitations in capturing complex image features compared to Deep Learning models (CNNs).
