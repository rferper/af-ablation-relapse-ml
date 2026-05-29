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

The main analysis (`final.ipynb`) performs a systematic grid search over:

| Axis | Values tested |
|---|---|
| Classifiers | Random Forest, XGBoost |
| Feature selection | Mutual Information (k = 15, 17, 20) |
| Imbalance handling | None, SMOTE-Tomek, Random Under-Sampling |
| Hyperparameter search | GridSearchCV, Repeated Stratified K-Fold (5×10) |

Metrics reported per configuration: Accuracy, Precision, Recall, F1, AUC, Balanced Accuracy (mean ± SD on training folds, hold-out test set).

## Repository structure

```
.
├── data/
│   ├── synthetic_data.csv                 # Public replacement for private dataset
│   ├── clean_dictionary_non_diabetic.csv  # Variable metadata (no patient data)
│   └── generate_synthetic_data.py         # Script to regenerate synthetic data
├── final.ipynb                            # Main grid search + SHAP analysis
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
