"""Gap rule: long intervals between recovered readings are gaps only if the
provider series is missing or irregular inside them (or they exceed 60 min)."""
import numpy as np
import pandas as pd

from cgmeal.io import native_readings
from cgmeal.meals import meal_response

from conftest import make_minute_frame


def _knots(minutes, period, flat_from, flat_to, level=110.0, slope=0.0, seed=0):
    """Knot values with noise, except a collinear stretch [flat_from, flat_to]."""
    rng = np.random.default_rng(seed)
    k = np.arange(0, minutes, period)
    v = 100 + rng.integers(-15, 15, len(k)).astype(float)
    m = (k >= flat_from) & (k <= flat_to)
    v[m] = level + slope * (k[m] - flat_from)
    return v


def _frame(knot_values, minutes=480, period=5):
    ts, kt, kv = make_minute_frame(minutes=minutes, native_period=period, glucose_fn=knot_values)
    return ts


def test_flat_stretch_on_regular_grid_is_used():
    ts = _frame(_knots(480, 5, 150, 185))  # 35 min flat, inside the meal window
    nat = native_readings(ts["dexcom"])
    t0 = ts.index[120]
    r = meal_response(nat, t0, "dexcom")
    assert r["ok"] and r["n_long_flat"] == 1 and r["n_long_sloped"] == 0


def test_sloped_collinear_stretch_is_used():
    ts = _frame(_knots(480, 5, 150, 175, level=110, slope=0.4))  # +2 mg/dL per 5 min
    nat = native_readings(ts["dexcom"])
    r = meal_response(nat, ts.index[120], "dexcom")
    assert r["ok"] and r["n_long_sloped"] == 1


def test_missing_value_inside_interval_is_a_gap():
    ts = _frame(_knots(480, 5, 150, 185))
    ts.iloc[155:176, ts.columns.get_loc("dexcom")] = np.nan  # 21 missing minutes
    nat = native_readings(ts["dexcom"])
    r = meal_response(nat, ts.index[120], "dexcom")
    assert not r["ok"] and r["reason"] == "cgm_gap_in_window"


def test_irregular_timestamps_inside_interval_are_a_gap():
    # provider inserts rows 3 min apart across a missed reading (as in the Libre files)
    ts = _frame(_knots(480, 5, 150, 185))
    drop = ts.index[[151, 152, 154, 155, 157, 158]]
    ts = ts.drop(drop)
    nat = native_readings(ts["dexcom"])
    assert not nat.loc[(nat["time"] >= ts.index[0] + pd.Timedelta(minutes=145))
                       & (nat["time"] < ts.index[0] + pd.Timedelta(minutes=185)), "regular_after"].all()
    r = meal_response(nat, ts.index[120], "dexcom")
    assert not r["ok"] and r["reason"] == "cgm_gap_in_window"


def test_collinear_stretch_longer_than_cap_is_a_gap():
    ts = _frame(_knots(480, 5, 130, 200))  # 70 min flat > 60-min cap
    nat = native_readings(ts["dexcom"])
    r = meal_response(nat, ts.index[120], "dexcom")
    assert not r["ok"] and r["reason"] == "cgm_gap_in_window"


def test_regular_after_flags():
    ts = _frame(_knots(120, 5, 200, 200), minutes=120)
    ts.iloc[50:55, ts.columns.get_loc("dexcom")] = np.nan
    nat = native_readings(ts["dexcom"])
    before = nat[nat["time"] < ts.index[50]]
    # the last reading before the missing block is not followed by a regular interval
    assert not before["regular_after"].iloc[-1]
    assert before["regular_after"].iloc[:-1].all()
    assert not nat["regular_after"].iloc[-1]
