"""
Generates synthetic data with the same schema as the private clinical dataset.
The synthetic data preserves column names, types, class imbalance ratio (84:30),
and medically plausible value distributions for AF ablation patients.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 114
n_neg, n_pos = 84, 30  # class 0 (no late relapse), class 1 (late relapse)

def bernoulli(p, size):
    return rng.binomial(1, p, size).astype(int)

def clipped_normal(mean, sd, low, high, size):
    return np.clip(rng.normal(mean, sd, size), low, high)

binary_cols = {
    "sex":                  bernoulli(0.68, n),
    "early_relapse":        bernoulli(0.22, n),
    "treatment":            bernoulli(0.55, n),
    "af_type":              bernoulli(0.62, n),
    "ehr":                  bernoulli(0.20, n),
    "ischemic":             bernoulli(0.25, n),
    "hypertension":         bernoulli(0.60, n),
    "thyroid":              bernoulli(0.20, n),
    "dyslipidemia":         bernoulli(0.40, n),
    "heart_failure":        bernoulli(0.10, n),
    "antiarrythmic_drugs":  bernoulli(0.50, n),
    "lba":                  bernoulli(0.10, n),
    "statina":              bernoulli(0.42, n),
    "acei":                 bernoulli(0.30, n),
    "sartan":               bernoulli(0.20, n),
    "acei_sartan":          bernoulli(0.45, n),
    "ca_bloker":            bernoulli(0.25, n),
    "diuretics":            bernoulli(0.30, n),
    "smoker":               bernoulli(0.15, n),
    "cha2ds2_vasc_scale":   bernoulli(0.50, n),
    "has_bled_scale":       bernoulli(0.15, n),
}

numerical_cols = {
    "age":                   clipped_normal(60,  9,   35,  85, n),
    "rdw_ba":                clipped_normal(13.2, 1.0, 10, 18,  n),
    "rdw_aa":                clipped_normal(13.1, 1.0, 10, 18,  n),
    "neutrophils_ba":        clipped_normal(4.5, 1.4,  1.2, 10, n),
    "neutrophils_aa":        clipped_normal(5.8, 1.8,  1.0, 14, n),
    "lymphocytes_ba":        clipped_normal(1.8, 0.6,  0.5,  5, n),
    "lymphocytes_aa":        clipped_normal(1.4, 0.6,  0.3,  5, n),
    "monocyte_ba":           clipped_normal(0.5, 0.15, 0.1,  1.5, n),
    "monocyte_aa":           clipped_normal(0.6, 0.18, 0.1,  1.8, n),
    "plt_ba":                clipped_normal(220, 55,   80, 500, n),
    "plt_aa":                clipped_normal(215, 60,   80, 500, n),
    "nlr_ba":                clipped_normal(2.7, 1.1,  0.5, 10, n),
    "nlr_aa":                clipped_normal(4.5, 2.0,  0.5, 15, n),
    "plr_ba":                clipped_normal(130, 45,   40, 400, n),
    "plr_aa":                clipped_normal(170, 65,   40, 500, n),
    "mlr_ba":                clipped_normal(0.30, 0.10, 0.05, 0.9, n),
    "mlr_aa":                clipped_normal(0.46, 0.16, 0.05, 1.2, n),
    "sii_ba":                clipped_normal(550, 250,  80, 2500, n),
    "sii_aa":                clipped_normal(850, 400, 100, 4000, n),
    "siri_ba":               clipped_normal(0.65, 0.30, 0.05, 3.0, n),
    "siri_aa":               clipped_normal(1.10, 0.55, 0.05, 5.0, n),
    "nhr_ba":                clipped_normal(3.5, 1.5,  0.5, 12,  n),
    "nhr_aa":                clipped_normal(5.5, 2.5,  0.5, 18,  n),
    "crp_hdl_ba":            clipped_normal(0.18, 0.14, 0.01, 1.0, n),
    "crp_hdl_aa":            clipped_normal(0.28, 0.22, 0.01, 1.5, n),
    "mhr_ba":                clipped_normal(0.38, 0.15, 0.05, 1.0, n),
    "mhr_aa":                clipped_normal(0.55, 0.22, 0.05, 1.5, n),
    "crp_ba":                clipped_normal(3.5, 3.5,   0.1, 30,  n),
    "crp_aa":                clipped_normal(8.0, 7.0,   0.1, 60,  n),
    "hdl":                   clipped_normal(52,  14,    20,  100, n),
    "body_weight":           clipped_normal(84,  14,    50,  140, n),
    "bmi":                   clipped_normal(28.5, 4.5,  17,  48,  n),
    "wc":                    clipped_normal(96,  12,    65,  140, n),
    "adiponectin":           clipped_normal(8.5, 5.0,   1,   40,  n),
    "leptin":                clipped_normal(14,  10,    1,   60,  n),
    "lar":                   clipped_normal(2.2, 1.8,   0.1, 12,  n),
    "t_pa_ba":               clipped_normal(9,   5,     1,   40,  n),
    "t_pa_aa":               clipped_normal(11,  6,     1,   50,  n),
    "pai1_ba":               clipped_normal(40,  20,    5,  120,  n),
    "pai1_aa":               clipped_normal(45,  22,    5,  130,  n),
    "scd40l_ba":             clipped_normal(1200, 700,  100, 6000, n),
    "scd40l_aa":             clipped_normal(1400, 800,  100, 7000, n),
    "btgcxcl7_ba":           clipped_normal(1800, 900,  200, 8000, n),
    "btgcxcl7_aa":           clipped_normal(1900, 950,  200, 8500, n),
    "fibrinogen_ba":         clipped_normal(3.2, 0.8,   1.0, 6.5, n),
    "fibrinogen_aa":         clipped_normal(3.5, 0.9,   1.0, 7.0, n),
    "d_dimery_ba":           clipped_normal(0.5, 0.35,  0.1, 3.5, n),
    "d_dimery_aa":           clipped_normal(0.6, 0.4,   0.1, 4.0, n),
    "svcam1_ba":             clipped_normal(800, 280,  200, 2500, n),
    "svcam1_aa":             clipped_normal(850, 300,  200, 2600, n),
    "slcam1_ba":             clipped_normal(350, 120,   80, 900,  n),
    "slcam1_aa":             clipped_normal(360, 130,   80, 950,  n),
    "hsil6_ba":              clipped_normal(4.5, 4.0,   0.5, 35,  n),
    "hsil6_aa":              clipped_normal(7.0, 6.5,   0.5, 55,  n),
    "pentraxin_ba":          clipped_normal(5.0, 3.5,   0.5, 25,  n),
    "pentraxin_aa":          clipped_normal(6.5, 4.5,   0.5, 30,  n),
    "wbc_ba":                clipped_normal(7.0, 1.8,   3.0, 14,  n),
    "wbc_aa":                clipped_normal(8.5, 2.2,   3.0, 18,  n),
    "vwf_ba":                clipped_normal(130, 40,    50, 300,  n),
    "vwf_aa":                clipped_normal(140, 45,    50, 320,  n),
    "tm_ba":                 clipped_normal(3.5, 1.5,   0.5, 10,  n),
    "tm_aa":                 clipped_normal(3.8, 1.6,   0.5, 11,  n),
    "troponin_ba":           clipped_normal(0.05, 0.04, 0.005, 0.4, n),
    "troponin_aa":           clipped_normal(0.12, 0.10, 0.005, 0.8, n),
    "cpk_ba":                clipped_normal(100, 55,    20, 400,  n),
    "cpk_aa":                clipped_normal(130, 70,    20, 500,  n),
    "ck_mb_ba":              clipped_normal(8.0, 4.5,   1,   40,  n),
    "ck_mb_aa":              clipped_normal(10.5, 6.0,  1,   55,  n),
    "st2_ba":                clipped_normal(22,  8,     8,   60,  n),
    "st2_aa":                clipped_normal(24,  9,     8,   65,  n),
    "lavolume_ba":           clipped_normal(105, 28,    40, 220,  n),
    "lavi":                  clipped_normal(48,  14,    18,  100, n),
    "ef":                    clipped_normal(58,  8,     30,  80,  n),
    "rbc_ba":                clipped_normal(4.7, 0.55,  3.0,  6.5, n),
    "hb_ba":                 clipped_normal(14.0, 1.6,  8,   18,  n),
    "hct_ba":                clipped_normal(42.5, 4.5,  25,  55,  n),
    "mcv_ba":                clipped_normal(88,  7,     68, 108,  n),
    "mch_ba":                clipped_normal(29,  3,     18,  38,  n),
    "mchc_ba":               clipped_normal(335, 12,   300, 370,  n),
    "na_ba":                 clipped_normal(139, 3,    128, 148,  n),
    "k_ba":                  clipped_normal(4.3, 0.4,   3.0,  5.8, n),
    "glucose_ba":            clipped_normal(95,  14,    60,  160, n),
    "cholesterol_ba":        clipped_normal(5.2, 1.1,   2.5,  9.0, n),
    "hdl_ba":                clipped_normal(1.35, 0.38, 0.5,  2.8, n),
    "ldl_ba":                clipped_normal(3.1, 0.95,  1.0,  6.5, n),
    "tg_ba":                 clipped_normal(1.6, 0.75,  0.4,  5.0, n),
    "tsh_ba":                clipped_normal(1.8, 1.2,   0.1,  8.0, n),
    "urea_ba":               clipped_normal(5.8, 1.8,   2.0, 14,  n),
    "creatinine_ba":         clipped_normal(88,  20,    45, 180,  n),
    "gfr_ba":                clipped_normal(78,  18,    25, 120,  n),
    "alat_ba":               clipped_normal(28,  16,    8,  120,  n),
    "aspat_ba":              clipped_normal(24,  10,    8,   80,  n),
    "rrsyst_ba":             clipped_normal(138, 18,    90, 200,  n),
    "rrsyst_aa":             clipped_normal(132, 16,    90, 190,  n),
    "rrdiast_ba":            clipped_normal(82,  11,    50, 120,  n),
    "rrdiast_aa":            clipped_normal(78,  10,    50, 115,  n),
    "treatment_time":        clipped_normal(120, 35,    45, 260,  n),
    "xray_copy_time":        clipped_normal(14,  6,     3,   45,  n),
    "cryoapplication_time":  clipped_normal(32,  9,     8,   65,  n),
    "number_applications":   np.round(clipped_normal(8.2, 2.2, 3, 16, n)).astype(int),
}

# Assemble DataFrame in original column order
col_order = [
    "sex", "age", "early_relapse", "treatment",
    "rdw_ba", "rdw_aa", "neutrophils_ba", "neutrophils_aa",
    "lymphocytes_ba", "lymphocytes_aa", "monocyte_ba", "monocyte_aa",
    "plt_ba", "plt_aa", "nlr_ba", "nlr_aa", "plr_ba", "plr_aa",
    "mlr_ba", "mlr_aa", "sii_ba", "sii_aa", "siri_ba", "siri_aa",
    "nhr_ba", "nhr_aa", "crp_hdl_ba", "crp_hdl_aa", "mhr_ba", "mhr_aa",
    "crp_ba", "crp_aa", "hdl", "body_weight", "bmi", "wc",
    "adiponectin", "leptin", "lar",
    "t_pa_ba", "t_pa_aa", "pai1_ba", "pai1_aa",
    "scd40l_ba", "scd40l_aa", "btgcxcl7_ba", "btgcxcl7_aa",
    "fibrinogen_ba", "fibrinogen_aa", "d_dimery_ba", "d_dimery_aa",
    "svcam1_ba", "svcam1_aa", "slcam1_ba", "slcam1_aa",
    "hsil6_ba", "hsil6_aa", "pentraxin_ba", "pentraxin_aa",
    "wbc_ba", "wbc_aa", "vwf_ba", "vwf_aa", "tm_ba", "tm_aa",
    "troponin_ba", "troponin_aa", "cpk_ba", "cpk_aa", "ck_mb_ba", "ck_mb_aa",
    "st2_ba", "st2_aa", "lavolume_ba", "lavi", "ef",
    "rbc_ba", "hb_ba", "hct_ba", "mcv_ba", "mch_ba", "mchc_ba",
    "na_ba", "k_ba", "glucose_ba", "cholesterol_ba", "hdl_ba", "ldl_ba",
    "tg_ba", "tsh_ba", "urea_ba", "creatinine_ba", "gfr_ba", "alat_ba", "aspat_ba",
    "rrsyst_ba", "rrsyst_aa", "rrdiast_ba", "rrdiast_aa",
    "af_type", "ehr", "ischemic", "hypertension", "thyroid", "dyslipidemia",
    "heart_failure", "antiarrythmic_drugs", "lba", "statina", "acei", "sartan",
    "acei_sartan", "ca_bloker", "diuretics", "smoker",
    "cha2ds2_vasc_scale", "has_bled_scale",
    "treatment_time", "xray_copy_time", "cryoapplication_time", "number_applications",
    "late_relapse",
]

all_data = {**binary_cols, **numerical_cols}
# Assign target: first n_neg rows = 0, last n_pos rows = 1, then shuffle
late_relapse = np.array([0] * n_neg + [1] * n_pos)
rng.shuffle(late_relapse)
all_data["late_relapse"] = late_relapse

df = pd.DataFrame(all_data)[col_order]

# Round numerical columns to 2 decimal places for readability
float_cols = [c for c in df.columns if c not in binary_cols and c != "number_applications" and c != "late_relapse"]
df[float_cols] = df[float_cols].round(2)

df.to_csv("synthetic_data.csv", index=False)
print(f"Saved synthetic_data.csv — shape: {df.shape}")
print(f"late_relapse distribution: {df['late_relapse'].value_counts().to_dict()}")
