"""Meal-response prediction with modality ablation.

Evaluation settings
-------------------
(a) Leave-one-subject-out (LOSO): each participant is predicted by a model
    trained on all other participants.  Tests a *new person*.
(b) Within-person temporal split: for every participant, the earliest
    ``TRAIN_FRAC`` of their eligible meals (by time) go to training and the
    later ones to testing; one model is trained on all participants' early
    meals.  Tests *a known person's future meals*.

No hyper-parameters are chosen on test data: gradient boosting uses fixed
settings chosen before running; ridge selects its penalty by inner
GroupKFold (grouped by participant) on the training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_FRAC = 0.7
RIDGE_ALPHAS = np.logspace(-2, 4, 13)
GBM_PARAMS = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                  l2_regularization=1.0, random_state=0)
MIN_TEST_MEALS = 5  # per-participant metrics only for participants with >= this many test meals


def make_model(name: str):
    if name == "ridge":
        pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), Ridge())
        return ("ridge", pipe)
    if name == "gbm":
        return ("gbm", HistGradientBoostingRegressor(**GBM_PARAMS))
    raise ValueError(name)


def fit_predict(name: str, Xtr, ytr, gtr, Xte) -> np.ndarray:
    kind, model = make_model(name)
    if kind == "ridge":
        n_splits = min(5, len(np.unique(gtr)))
        gs = GridSearchCV(model, {"ridge__alpha": RIDGE_ALPHAS}, cv=GroupKFold(n_splits=n_splits),
                          scoring="neg_mean_squared_error")
        gs.fit(Xtr, ytr, groups=gtr)
        return gs.predict(Xte)
    model.fit(Xtr, ytr)
    return model.predict(Xte)


def loso_splits(groups):
    """Yield (train_idx, test_idx) with exactly one participant held out."""
    groups = np.asarray(groups)
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        yield tr, te


def temporal_split(df: pd.DataFrame, frac: float = TRAIN_FRAC, pid_col="pid", time_col="time") -> np.ndarray:
    """Boolean train mask: the earliest ``frac`` of each participant's meals.

    Every training meal of a participant precedes every test meal of that
    participant.  Participants with < 2 meals go entirely to training.
    """
    train = np.zeros(len(df), bool)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for _, sub in df.groupby(pid_col):
        order = pos[sub.sort_values(time_col).index].to_numpy()
        n_tr = int(np.floor(frac * len(order))) if len(order) >= 2 else len(order)
        n_tr = max(n_tr, 1)
        train[order[:n_tr]] = True
    return train


def personal_history(y, groups, train_mask) -> np.ndarray:
    """Participant's mean training outcome (leave-one-out for training rows).

    Uses training rows only, so no test outcome ever enters a feature.
    """
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr = (groups == g) & train_mask
        te = (groups == g) & ~train_mask
        s, n = y[tr].sum(), tr.sum()
        if n > 1:
            out[tr] = (s - y[tr]) / (n - 1)
        if n > 0:
            out[te] = s / n
    return out


def metrics(y, yhat, groups) -> dict:
    y, yhat, groups = np.asarray(y, float), np.asarray(yhat, float), np.asarray(groups)
    per = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() >= MIN_TEST_MEALS:
            r = stats.pearsonr(y[m], yhat[m])[0] if np.std(yhat[m]) > 0 and np.std(y[m]) > 0 else np.nan
            per.append({"r": r, "r2": r2_score(y[m], yhat[m]), "mae": mean_absolute_error(y[m], yhat[m])})
    per = pd.DataFrame(per)
    out = {"n_meals": int(len(y)), "n_participants": int(len(np.unique(groups))),
           "pooled_r": float(stats.pearsonr(y, yhat)[0]), "pooled_r2": float(r2_score(y, yhat)),
           "pooled_mae": float(mean_absolute_error(y, yhat)),
           "n_participants_metrics": int(len(per))}
    for k in ["r", "r2", "mae"]:
        v = per[k].dropna() if len(per) else pd.Series(dtype=float)
        out[f"pp_{k}_median"] = float(v.median()) if len(v) else np.nan
        out[f"pp_{k}_q1"] = float(v.quantile(0.25)) if len(v) else np.nan
        out[f"pp_{k}_q3"] = float(v.quantile(0.75)) if len(v) else np.nan
    return out


def run_loso(X: pd.DataFrame, y, groups, model: str) -> np.ndarray:
    yhat = np.full(len(y), np.nan)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    for tr, te in loso_splits(groups):
        yhat[te] = fit_predict(model, X.iloc[tr], y[tr], groups[tr], X.iloc[te])
    return yhat


def run_temporal(X: pd.DataFrame, y, groups, train_mask, model: str) -> np.ndarray:
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    yhat = np.full(len(y), np.nan)
    yhat[~train_mask] = fit_predict(model, X[train_mask], y[train_mask], groups[train_mask], X[~train_mask])
    return yhat
