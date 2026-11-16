import numpy as np
import pandas as pd
import pytest

from cgmeal.io import Participant, native_readings


def make_minute_frame(start="2020-01-01 06:00", minutes=600, native_period=5, glucose_fn=None, seed=0):
    """Synthetic 1-min frame whose glucose is linearly interpolated from native readings."""
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=minutes, freq="1min")
    knots = np.arange(0, minutes, native_period)
    if glucose_fn is None:
        kv = 100 + rng.normal(0, 8, len(knots)).round()
    elif callable(glucose_fn):
        kv = np.array([glucose_fn(k) for k in knots], float)
    else:  # explicit knot values
        kv = np.asarray(glucose_fn, float)
    g = np.interp(np.arange(minutes), knots, kv)
    g[np.arange(minutes) > knots[-1]] = np.nan
    ts = pd.DataFrame(index=pd.DatetimeIndex(t, name="time"))
    ts["dexcom"] = g
    ts["libre"] = g
    ts["hr"] = 70 + rng.normal(0, 3, minutes)
    ts["activity_kcal"] = 1.2
    ts["mets"] = 1.0
    ts["intensity"] = np.nan
    return ts, t[knots], kv


def native_from_knots(times, values):
    return pd.DataFrame({"time": pd.DatetimeIndex(times), "glucose": np.asarray(values, float)})


def make_participant(ts, meal_times, pid=1, carbs=50.0):
    meals = pd.DataFrame({"time": pd.to_datetime(meal_times), "meal_type": "lunch",
                          "kcal": 400.0, "carbs": carbs, "protein": 20.0, "fat": 10.0, "fiber": 3.0,
                          "amount_consumed": 100.0, "has_photo": True})
    native = {d: native_readings(ts[d]) for d in ["dexcom", "libre"]}
    return Participant(pid=pid, ts=ts, meals=meals, native=native)


@pytest.fixture
def synthetic():
    return make_minute_frame
