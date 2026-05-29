# AF Ablation: Late Relapse Prediction with ML

Binary classification study predicting **late relapse** (3–12 months post-ablation) in non-diabetic patients undergoing atrial fibrillation (AF) catheter ablation.

## Problem

AF ablation has a significant recurrence rate. Identifying pre-procedure biomarkers that predict late relapse can help clinicians select patients and plan follow-up. The dataset contains 120+ clinical features collected before and after the ablation procedure, with a 2.8:1 class imbalance (no-relapse vs. relapse).

## Dataset

The original clinical dataset is private (patient-level data from a hospital registry). This repository uses **synthetic data** (`data/synthetic_data.csv`) generated to match:
- The original schema (122 columns, same variable names and types)
- The class distribution (84 class-0 / 30 class-1, ratio ≈ 2.8:1)
- Medically plausible distributions for all biomarkers

The synthetic data is reproducible — see `data/generate_synthetic_data.py`.

## Feature groups

| Group | Examples |
|---|---|
| Demographics | age, sex, BMI, waist circumference |
| Haematology (pre/post) | RDW, WBC, neutrophils, lymphocytes, platelets |
| Inflammatory indices (pre/post) | NLR, PLR, SII, SIRI, CRP, IL-6, pentraxin |
| Coagulation (pre/post) | fibrinogen, D-dimer, vWF, PAI-1, t-PA |
| Echocardiography | LA volume, LAVI, ejection fraction |
| Comorbidities & medications | hypertension, heart failure, statins, ACEi/ARBs, … |
| Procedure | treatment type, duration, number of cryoapplications |

## Methods

`final.ipynb` runs a systematic grid search followed by explainability analysis:

### 1. Preprocessing
- High-correlation filter (Pearson r > 0.95 → drop)
- StandardScaler on continuous features only

### 2. Broad grid search

| Axis | Values tested |
|---|---|
| **Classifiers** | Logistic Regression, Random Forest, XGBoost, SVM |
| **Feature selection** | None (all features), RFE (L1-LR), Mutual Information (k = 10, 15, 20, 25) |
| **Imbalance handling** | None, Random Under-Sampling, SMOTE, SMOTE-Tomek, ADASYN, BorderlineSMOTE |
| **Hyperparameter search** | GridSearchCV, Repeated Stratified K-Fold (5 splits × 10 repeats) |

Metrics per configuration: Accuracy, Precision, Recall, F1, AUC, Balanced Accuracy (mean ± SD on training folds + hold-out test set).

### 3. Focused analysis

Best-performing configuration (RF + Mutual Information k=15 + Random Under-Sampling) re-evaluated in detail with tree feature importances for Logistic Regression (coefficients), Random Forest (Gini importance), and XGBoost (gain).

### 4. TabPFN

- **TabPFN feature selection** — uses `tabpfn_extensions` sequential feature selection (TabPFN attention as the scorer) to rank and select the top-15 features; a Random Forest is then trained on the selected subset.
- **AutoTabPFNClassifier** — end-to-end AutoML wrapper over TabPFN ensembles, evaluated on the same split and sampling as the classical models.

### 5. Explainability (SHAP)

SHAP values computed for both the best RF and XGBoost configurations:
- Global feature importance (beeswarm summary plot)
- Per-prediction explanation (waterfall plot for individual patients)

## Repository structure

```
.
├── data/
│   ├── synthetic_data.csv                 # Public replacement for private dataset
│   ├── clean_dictionary_non_diabetic.csv  # Variable metadata (no patient data)
│   └── generate_synthetic_data.py         # Script to regenerate synthetic data
├── final.ipynb                            # Main analysis: grid search + TabPFN + SHAP
├── robustness.ipynb                       # Stability analysis across 10 random splits
├── results.csv                            # Example results from the main analysis
├── results_robustness_split.csv           # Per-split robustness results
├── results_robustness_average.csv         # Averaged robustness results
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook final.ipynb
```

Tested with Python 3.10–3.12.

## Dependencies

| Package | Purpose |
|---|---|
| scikit-learn ≥ 1.8 | Classical classifiers, feature selection, cross-validation |
| imbalanced-learn ≥ 0.14 | SMOTE, Random Under-Sampling, and variants |
| xgboost ≥ 2.0 | Gradient boosting classifier |
| shap ≥ 0.46 | SHAP explainability (TreeExplainer, beeswarm, waterfall) |
| tabpfn ≥ 8.0 | TabPFN in-context learning classifier |
| tabpfn-extensions ≥ 0.4 | AutoTabPFN and TabPFN-based feature selection |
| pandas, numpy, matplotlib, seaborn | Data handling and visualisation |
