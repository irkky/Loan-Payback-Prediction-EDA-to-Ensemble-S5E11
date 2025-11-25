# 🏆 Predicting Loan Payback  
### Playground Series — Season 5, Episode 11

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)

> **Author:** Rishabh Kumar Kannaujiya  
> **Date:** Nov 2025  

A machine learning solution to predict the probability of a borrower paying back a loan using a weighted ensemble of Gradient Boosting models.

---

## 📑 Table of Contents
- [📍 Overview](#-overview)
- [⚙️ Tech Stack](#️-tech-stack)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Preprocessing Pipeline](#️-preprocessing-pipeline)
- [🤖 Modeling Strategy](#-modeling-strategy)
- [🏆 Results & Performance](#-results--performance)
- [🚀 How to Run](#-how-to-run)

---

## 📍 Overview

This project solves a **Binary Classification** task — **predicting whether a customer will successfully pay back a loan**.

The final approach uses a **weighted ensemble** of three powerful Gradient Boosting algorithms:  
**XGBoost**, **LightGBM**, and **CatBoost**, optimized to maximize the ROC AUC score.

---

## ⚙️ Tech Stack

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23F44900.svg?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-%2300A65C.svg?style=flat&logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-%23FFBB00.svg?style=flat&logo=catboost&logoColor=white)

---

## 📊 Exploratory Data Analysis

We analyzed key dataset properties to identify correlations, missing values, and distribution patterns.

| Feature | Observation |
|--------|-------------|
| **Target** | `loan_paid_back` is imbalanced (~80% Paid, ~20% Default). |
| **Correlations** | Numerical correlations analyzed via heatmap to detect redundancy. |
| **Missing Values** | Present in both numerical & categorical features. |

<details>
<summary><strong>Click to view Feature Correlation Code</strong></summary>

```python
# Selecting only numerical columns for correlation
numerical_data = train_df.select_dtypes(include=[np.number])
corr = numerical_data.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
```
</details>

---

## 🛠️ Preprocessing Pipeline

The following preprocessing steps were applied:

1. **Cleaning**
   - Dropped `id` column (non-informative).

2. **Imputation**
   - **Numerical:** Filled missing values with **Median**.
   - **Categorical:** Filled missing values with the **Mode**.

3. **Encoding**
   - Applied **Label Encoding** to categorical features  
     (`gender`, `marital_status`, `education_level`, etc.)  
     — ideal for tree-based models.

---

## 🤖 Modeling Strategy

A “Power Trio” of Gradient Boosting models was used, each trained with **Early Stopping**.

### 1. XGBoost
- `n_estimators=2000`, `learning_rate=0.02`, `max_depth=6`
- Regularization via `subsample=0.8`, `colsample_bytree=0.8`

### 2. LightGBM
- `n_estimators=2000`, `learning_rate=0.02`, `num_leaves=31`
- Fast & resource efficient

### 3. CatBoost
- `iterations=2000`, `learning_rate=0.02`, `depth=6`
- Native handling of categorical features

### 4. Weighted Ensemble

Instead of using a simple average, weights were assigned based on validation ROC AUC:

```python
# Final Ensemble Weights
weighted_preds = (LGBM_preds * 0.70) + (XGB_preds * 0.15) + (CatBoost_preds * 0.15)
```

---

## 🏆 Results & Performance

Evaluation Metric: **ROC AUC**  
Train/validation split: **80/20**

| Model | ROC AUC (Validation) |
|-------|------------------------|
| **CatBoost** | 0.92051 |
| **XGBoost** | 0.92096 |
| **LightGBM** | **0.92200** |
| **Ensemble** | **0.92203 🚀** |

> **Conclusion:**  
> LightGBM performs best individually, while the weighted ensemble provides the most stable and highest overall score.

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/loan-payback-prediction.git
cd loan-payback-prediction
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost seaborn matplotlib
```

### 3. Download Dataset

Place `train.csv` and `test.csv` from Kaggle into the `input/` directory.

### 4. Run the Notebook

Open `main.ipynb` in **Jupyter Lab**, **VS Code**, or **Kaggle Notebook**, then **Run All**.

---

<div align="center">

⭐ **If you find this project useful, please consider giving it a star!** ⭐

</div>
