"""Secondary task: can participant-level wearable or CGM features separate groups?

Groups are *defined* from screening HbA1c, so lab features (HbA1c, fasting
glucose, ...) would make this task circular and are excluded.  CGM-derived
features remain partly circular: HbA1c reflects average glycaemia over ~3
months, which mean CGM glucose estimates directly (this is the basis of the
glucose management indicator).  Wearable-only features are not circular in
this sense but may be confounded by age and BMI, which differ between groups.

Model: multinomial L2 logistic regression with median imputation and
standardisation, fixed C = 1.0 (no tuning), evaluated by leave-one-participant-
out.  Chance level is assessed by a label-permutation test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WEARABLE_FEATURES = ["hr_median", "hr_p05", "hr_sd", "activity_kcal_per_min", "mets_mean", "hr_night_median"]
CGM_FEATURES = ["mean_glucose", "sd_glucose", "cv_pct", "tir_70_180_pct", "titr_70_140_pct", "tar_180_pct",
                "mage", "median_iauc", "median_peak_rise", "median_time_to_peak"]
LABELS = ["healthy", "prediabetes", "t2d"]


def wearable_features(ts: pd.DataFrame) -> dict:
    hr = ts["hr"].dropna()
    night = hr[(hr.index.hour >= 0) & (hr.index.hour < 5)]
    return {
        "hr_median": hr.median(), "hr_p05": hr.quantile(0.05), "hr_sd": hr.std(),
        "activity_kcal_per_min": ts["activity_kcal"].mean(), "mets_mean": ts["mets"].mean(),
        "hr_night_median": night.median() if len(night) else np.nan,
    }


def _model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=5000))


def loo_predict(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        m = _model().fit(X[tr], y[tr])
        pred[i] = m.predict(X.iloc[[i]])[0]
    return pred


def evaluate(X: pd.DataFrame, y, n_perm: int = 500, seed: int = 0) -> dict:
    y = np.asarray(y, dtype=object)
    pred = loo_predict(X, y)
    bacc = balanced_accuracy_score(y, pred)
    rng = np.random.default_rng(seed)
    null = np.array([balanced_accuracy_score(yp, loo_predict(X, yp))
                     for yp in (rng.permutation(y) for _ in range(n_perm))])
    cm = confusion_matrix(y, pred, labels=LABELS)
    return {"balanced_accuracy": bacc, "macro_f1": f1_score(y, pred, average="macro", labels=LABELS),
            "perm_p": (1 + np.sum(null >= bacc)) / (1 + n_perm), "null_mean": float(null.mean()),
            "n": int(len(y)), "confusion": cm}
