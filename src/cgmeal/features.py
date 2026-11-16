"""Per-meal predictors grouped by modality.

Every pre-meal feature uses only data with timestamp <= meal start:
* glucose context uses *native* readings at or before t0 (the provider's
  1-min interpolation would leak the next reading);
* activity uses the 1-min Fitbit rows in [t0 - 60, t0) (strictly before);
* time uses the meal clock time and the previous episode's start.
Post-meal activity ([t0, t0 + 120) min) is computed separately under the
``activity_post`` group and is used only in a clearly labelled secondary
analysis, because it is not known when the meal is eaten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVITY_PRE_MIN = 60
ACTIVITY_POST_MIN = 120
TREND_MIN = 30
SINCE_PREV_CAP_MIN = 720

FEATURE_GROUPS = {
    "meal": ["carbs", "protein", "fat", "fiber", "kcal"],
    "glucose": ["pre_glucose", "pre_trend", "pre_mean_120"],
    "activity": ["hr_pre", "activity_kcal_pre", "mets_pre"],
    "time": ["hour_sin", "hour_cos", "log_min_since_prev"],
    "person": ["age", "sex_male", "bmi", "hba1c", "fasting_glucose", "insulin", "triglycerides", "hdl"],
    "activity_post": ["hr_post", "activity_kcal_post", "mets_post"],
}

ABLATION_STEPS = [
    ("meal", ["meal"]),
    ("+ glucose context", ["meal", "glucose"]),
    ("+ activity", ["meal", "glucose", "activity"]),
    ("+ time", ["meal", "glucose", "activity", "time"]),
    ("+ person labs", ["meal", "glucose", "activity", "time", "person"]),
]


def columns_for(groups) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def _rel(times, t0) -> np.ndarray:
    return (pd.DatetimeIndex(times) - t0).total_seconds().to_numpy() / 60.0


def glucose_context(native: pd.DataFrame, t0: pd.Timestamp) -> dict:
    """Pre-meal glucose level and trend from native readings at or before t0."""
    rel = _rel(native["time"], t0)
    g = native["glucose"].to_numpy(float)
    past = rel <= 0
    out = {"pre_glucose": np.nan, "pre_trend": np.nan, "pre_mean_120": np.nan}
    w = past & (rel >= -TREND_MIN)
    if w.any():
        out["pre_glucose"] = float(g[w].mean())  # identical to the response baseline
    if w.sum() >= 2:
        out["pre_trend"] = float(np.polyfit(rel[w], g[w], 1)[0])  # mg/dL per min
    w2 = past & (rel >= -120)
    if w2.any():
        out["pre_mean_120"] = float(g[w2].mean())
    return out


def activity_window(ts: pd.DataFrame, t0: pd.Timestamp, start_min: float, end_min: float, suffix: str) -> dict:
    """Mean HR, mean activity kcal per minute and mean METs in [t0+start, t0+end)."""
    rel = _rel(ts.index, t0)
    w = (rel >= start_min) & (rel < end_min)
    sub = ts.loc[w]
    def mean(col):
        v = sub[col].to_numpy(float)
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    # activity kcal as a rate (kcal/min) so that windows with a few missing
    # minutes or timestamp jumps stay comparable
    return {
        f"hr_{suffix}": mean("hr"),
        f"activity_kcal_{suffix}": mean("activity_kcal"),
        f"mets_{suffix}": mean("mets"),
    }


def time_features(t0: pd.Timestamp, min_since_prev: float) -> dict:
    hour = t0.hour + t0.minute / 60
    since = SINCE_PREV_CAP_MIN if pd.isna(min_since_prev) else min(min_since_prev, SINCE_PREV_CAP_MIN)
    return {
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "log_min_since_prev": np.log(since),
    }


def build_features(part, ep: pd.DataFrame, people: pd.DataFrame, device: str = "dexcom") -> pd.DataFrame:
    """Feature table (one row per episode, same index as ``ep``)."""
    rows = []
    for _, e in ep.iterrows():
        t0 = e["time"]
        r = {c: e[c] for c in FEATURE_GROUPS["meal"]}
        r.update(glucose_context(part.native[device], t0))
        r.update(activity_window(part.ts, t0, -ACTIVITY_PRE_MIN, 0, "pre"))
        r.update(activity_window(part.ts, t0, 0, ACTIVITY_POST_MIN, "post"))
        r.update(time_features(t0, e["min_since_prev"]))
        rows.append(r)
    f = pd.DataFrame(rows, index=ep.index)
    person = people.loc[part.pid, FEATURE_GROUPS["person"]]
    for c in FEATURE_GROUPS["person"]:
        f[c] = float(person[c])
    return f
