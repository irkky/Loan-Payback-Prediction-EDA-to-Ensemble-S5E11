# 🏆 Predicting Loan Payback  
### Playground Series — Season 5, Episode 11

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle)

> **Author:** Rishabh Kumar Kannaujiya  
> **Date:** Nov 2025  

A machine learning solution to predict the probability of a borrower paying back a loan using a **stacked ensemble** of Gradient Boosting models with Logistic Regression as the meta-learner.

---

## 📑 Table of Contents
- [📍 Overview](#-overview)
- [⚙️ Tech Stack](#️-tech-stack)
- [📊 Exploratory Data Analysis](#-exploratory-data-analysis)
- [🛠️ Preprocessing Pipeline](#️-preprocessing-pipeline)
- [🔬 Feature Engineering](#-feature-engineering)
- [🤖 Modeling Strategy](#-modeling-strategy)
- [🏆 Results & Performance](#-results--performance)
- [🚀 How to Run](#-how-to-run)

---

## 📍 Overview

This project solves a **Binary Classification** task — **predicting whether a customer will successfully pay back a loan**.

The final approach uses a **stacked ensemble** architecture with:
- **Base Models:** XGBoost, LightGBM, and CatBoost (GPU-accelerated)
- **Meta-Learner:** Logistic Regression combining base model predictions
- **Cross-Validation:** 10-Fold Stratified CV for robust OOF predictions
- **Optimization Metric:** ROC AUC Score

---

## ⚙️ Tech Stack

![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23F44900.svg?style=flat&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-%2300A65C.svg?style=flat&logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-%23FFBB00.svg?style=flat&logo=catboost&logoColor=white)
![Category Encoders](https://img.shields.io/badge/Category_Encoders-blue.svg?style=flat)

**Key Libraries:**
- `category_encoders`: Advanced Target Encoding for categorical features
- `tqdm`: Progress tracking for cross-validation
- GPU acceleration for all three boosting algorithms

---

## 📊 Exploratory Data Analysis

Key dataset characteristics analyzed:

| Aspect | Finding |
|--------|---------|
| **Target Distribution** | `loan_paid_back` is imbalanced (~80% Paid, ~20% Default) |
| **Missing Values** | Present in both numerical and categorical features |
| **Feature Types** | Mix of numerical (income, credit score) and categorical (grade, employment status) |
| **Data Drift** | Adversarial validation performed to check Train-Test similarity |

---

## 🛠️ Preprocessing Pipeline

### 1. **Data Cleaning**
- Dropped non-informative `id` column
- Combined Train and Test sets for consistent preprocessing

### 2. **Missing Value Imputation**
- **Numerical Features:** Filled with median values
- **Categorical Features:** Filled with "Missing" label

### 3. **Encoding Strategy**
- **Target Encoding:** Applied using `category_encoders` with smoothing=20
  - Prevents data leakage by fitting on train fold only
  - Applied separately within each CV fold
- **CatBoost:** Uses native categorical handling (no encoding needed)

### 4. **Adversarial Validation**
- Trained classifier to distinguish Train vs Test samples
- **Pass Threshold:** AUC < 0.70 indicates similar distributions
- Helps identify potential data drift issues

---

## 🔬 Feature Engineering

Advanced features created to capture financial relationships:

### **Financial Ratios**
```python
loan_to_income = loan_amount / (annual_income + 1)
monthly_burden = loan_amount / (annual_income / 12 + 1)
interest_burden = loan_amount * (interest_rate / 100)
```

### **Log Transformations**
- `log_income = log1p(annual_income)`
- `log_loan = log1p(loan_amount)`
- Normalizes skewed monetary distributions

### **Binning**
- **Credit Score:** Binned into 5 categories (Poor, Fair, Good, Very Good, Excellent)
- **Loan Amount:** Quantile-based binning (10 bins)

### **GroupBy Aggregations**
- `loan_vs_grade_mean`: Loan amount relative to grade average
- `income_vs_job_mean`: Income relative to employment status average

---

## 🤖 Modeling Strategy

### **Architecture: Stacked Ensemble**

#### **Level 0: Base Models (10-Fold CV)**

**1. XGBoost**
```python
Parameters (Optimized via Hyperparameter Tuning):
- learning_rate: 0.063
- max_depth: 6
- subsample: 0.988
- colsample_bytree: 0.942
- n_estimators: 5000
- device: 'cuda' (GPU acceleration)
- tree_method: 'hist'
```

**2. LightGBM**
```python
Parameters:
- learning_rate: 0.118
- num_leaves: 110
- max_depth: 5
- subsample: 0.855
- colsample_bytree: 0.507
- device: 'gpu'
- n_estimators: 5000
```

**3. CatBoost**
```python
Parameters:
- learning_rate: 0.139
- depth: 5
- l2_leaf_reg: 0.522
- subsample: 0.652
- task_type: 'GPU'
- bootstrap_type: 'Bernoulli'
- iterations: 5000
- early_stopping_rounds: 100
```

#### **Level 1: Meta-Learner**

**Logistic Regression**
- Combines Out-of-Fold predictions from all base models
- Learns optimal weights for ensemble
- Prevents overfitting through CV-based training

### **Cross-Validation Strategy**
- **Method:** Stratified K-Fold (10 splits)
- **Purpose:** Generate robust OOF predictions for meta-model
- **Benefit:** Reduces overfitting and improves generalization

---

## 🏆 Results & Performance

**Evaluation Metric:** ROC AUC Score  
**Validation Strategy:** 10-Fold Stratified Cross-Validation

### Model Performance

Individual model performance varies by fold, with the final stacked ensemble providing the most robust predictions through Logistic Regression meta-learning.

**Key Features:**
- GPU acceleration significantly reduces training time
- Target encoding improves categorical feature utilization
- Stacking architecture captures complementary model strengths
- Post-processing clips predictions to [0.001, 0.999] range

---

## 🚀 How to Run

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/loan-payback-prediction.git
cd loan-payback-prediction
```

### 2. **Install Dependencies**

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost category-encoders seaborn matplotlib tqdm
```

### 3. **Download Dataset**

- Download competition data from [Kaggle Playground Series S5E11](https://www.kaggle.com/competitions/playground-series-s5e11)
- Place `train.csv`, `test.csv`, and `sample_submission.csv` in the `/kaggle/input/playground-series-s5e11/` directory

### 4. **Run the Notebook**

**Option A: Kaggle Notebook (Recommended)**
- Upload `main.ipynb` to Kaggle
- Enable GPU accelerator in notebook settings
- Run all cells

**Option B: Local Jupyter**
```bash
jupyter notebook main.ipynb
```
- Ensure GPU is available and CUDA is configured
- Update file paths if running locally

### 5. **Output**

The notebook generates:
- `submission.csv`: Final predictions for test set
- Correlation heatmap of model predictions
- Feature importance visualization
- Prediction distribution histogram

---

## 📈 Visualization Outputs

The notebook produces several analytical visualizations:

1. **Model Correlation Heatmap:** Shows diversity among base model predictions
2. **LightGBM Feature Importance:** Top 20 features ranked by gain
3. **Prediction Distribution:** Histogram of final probability predictions

---

## 🎯 Key Takeaways

- **Stacking > Single Models:** Meta-learner effectively combines model strengths
- **GPU Acceleration:** Critical for training large ensembles efficiently
- **Target Encoding:** Powerful technique for high-cardinality categorical features
- **Feature Engineering:** Financial ratios and aggregations boost predictive power
- **Cross-Validation:** 10-fold CV ensures robust OOF predictions for stacking

---

<div align="center">

⭐ **If you find this project useful, please consider giving it a star!** ⭐

**Built with 🧠 by [Rishabh Kumar Kannaujiya](https://github.com/yourusername)**

</div>
