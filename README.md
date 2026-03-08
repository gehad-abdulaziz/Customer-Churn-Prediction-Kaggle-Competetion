# Customer Churn Prediction
Kaggle Playground Series S6E3 — Predict the likelihood of customer churn using an ensemble of gradient boosting models.

---

## Overview

This project targets the [Kaggle Playground Series S6E3](https://kaggle.com/competitions/playground-series-s6e3) competition.
The goal is to predict customer churn probability, evaluated by **ROC-AUC** on the test set.

The solution combines **XGBoost**, **LightGBM**, and **CatBoost** in a 5-fold cross-validated ensemble with optimized blending weights.

---

## Repository Structure

```
.
├── customer-churn-improved.ipynb   # Main training notebook
├── submission.csv                  # Final predictions
└── README.md
```

---

## Dataset

| File | Description |
|------|-------------|
| `train.csv` | 594,194 rows — includes target column `Churn` |
| `test.csv` | Unlabeled rows for submission |
| `sample_submission.csv` | Expected submission format |

Features include customer demographics, subscription services, contract type, payment method, and billing charges.
Download from the [competition page](https://kaggle.com/competitions/playground-series-s6e3).

---

## Approach

### Feature Engineering

Raw features are preprocessed and enriched with domain-driven derived features:

| Feature | Description |
|---------|-------------|
| `ChargesPerMonth` | TotalCharges normalized by tenure |
| `ChargesDiff` | Difference between monthly and average charges |
| `ServiceCount` | Number of add-on services subscribed |
| `LowTenureHighCharge` | New customers with high monthly bills (strong churn signal) |
| `IsMonthToMonth` | Flag for high-risk contract type |
| `HasFiberOptic` | Fiber optic customers tend to churn more |
| `SeniorSingle` | Senior citizens without a partner |
| `TenureGroup` | Binned tenure into 4 segments |

### Models

Three gradient boosting models are trained independently inside a **5-fold StratifiedKFold**:

- **XGBoost** — `tree_method='hist'`, GPU accelerated, `scale_pos_weight` for class imbalance
- **LightGBM** — `num_leaves=63`, `class_weight='balanced'`, early stopping
- **CatBoost** — `auto_class_weights='Balanced'`, GPU accelerated, best model selection

Class imbalance is handled natively through each model's built-in weighting — no SMOTE or oversampling.

### Ensemble

Out-of-fold predictions from all three models are blended with weights optimized via `scipy.optimize.minimize` to maximize OOF ROC-AUC.

---

## Results

| Model | OOF ROC-AUC |
|-------|------------|
| XGBoost | ~0.845 |
| LightGBM | ~0.847 |
| CatBoost | ~0.848 |
| **Blended Ensemble** | **~0.851** |

---

## Setup

```bash
pip install xgboost lightgbm catboost scikit-learn pandas numpy scipy
```

Run the notebook on Kaggle with GPU enabled (Tesla T4) for best performance.

---

## Evaluation

Submissions are scored on **Area Under the ROC Curve (AUC)** between predicted probabilities and observed churn labels.

---

## Competition

| | |
|--|--|
| Series | Kaggle Playground Series — Season 6 Episode 3 |
| Start | March 1, 2026 |
| Deadline | March 31, 2026 |
| Metric | ROC-AUC |
| Prize | Kaggle merchandise (Top 3) |
