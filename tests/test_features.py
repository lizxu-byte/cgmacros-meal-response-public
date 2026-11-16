import numpy as np
import pandas as pd

from cgmeal.features import build_features, FEATURE_GROUPS
from cgmeal.io import native_readings

from conftest import make_minute_frame, make_participant


def _people():
    return pd.DataFrame({"age": [40.0], "sex_male": [1.0], "bmi": [25.0], "hba1c": [5.5],
                         "fasting_glucose": [90.0], "insulin": [5.0], "triglycerides": [100.0],
                         "hdl": [50.0]}, index=pd.Index([1], name="pid"))


def test_no_future_data_in_pre_meal_features():
    ts, kt, kv = make_minute_frame(minutes=600, native_period=15, seed=1)
    t0 = ts.index[307]  # between native readings at 300 and 315 min
    p1 = make_participant(ts, [t0])
    f1 = build_features(p1, _feature_ep(p1), _people(), device="libre")
    # the future differs: every native reading after t0 changes (the provider's
    # 1-min interpolation is rebuilt from them) and so do HR and activity
    rng = np.random.default_rng(7)
    kv2 = kv.copy()
    kv2[kt > t0] += rng.normal(80, 30, (kt > t0).sum()).round()
    ts2, _, _ = make_minute_frame(minutes=600, native_period=15, glucose_fn=kv2, seed=1)
    ts2[["hr", "activity_kcal", "mets"]] = ts[["hr", "activity_kcal", "mets"]]
    after = ts2.index > t0
    ts2.loc[after, "hr"] = 180
    ts2.loc[after, "activity_kcal"] = 15
    assert not np.isclose(ts.loc[t0, "libre"], ts2.loc[t0, "libre"])  # interpolated value at t0 leaks
    p2 = make_participant(ts2, [t0])
    f2 = build_features(p2, _feature_ep(p2), _people(), device="libre")
    pre_cols = [c for g in ["meal", "glucose", "activity", "time", "person"] for c in FEATURE_GROUPS[g]]
    pd.testing.assert_frame_equal(f1[pre_cols], f2[pre_cols])
    # post-meal features are allowed to (and do) change
    assert not np.isclose(f1["hr_post"].iloc[0], f2["hr_post"].iloc[0])


def _feature_ep(p):
    from cgmeal.meals import flag_overlap, merge_episodes
    return flag_overlap(merge_episodes(p.meals))
