import numpy as np
import pandas as pd

from cgmeal import meals
from cgmeal.meals import merge_episodes, flag_overlap, meal_response, positive_area, implausible

from conftest import make_minute_frame, make_participant, native_from_knots


def test_positive_area_triangle_and_trapezoid():
    # triangle 0 -> 60 -> 0 over 120 min: area = 0.5 * 120 * 60
    assert np.isclose(positive_area([0, 60, 120], [0, 60, 0]), 3600)
    # entirely below baseline contributes nothing
    assert positive_area([0, 60], [-5, -10]) == 0
    # crossing: from -10 to +10 over 20 min -> positive triangle 0.5 * 10 * 10
    assert np.isclose(positive_area([0, 20], [-10, 10]), 50)


def _response_for(fn, period=5, device="dexcom"):
    # native readings given directly (flat stretches have no detectable knots
    # in an interpolated series, see io.native_readings)
    _, kt, kv = make_minute_frame(minutes=400, native_period=period, glucose_fn=fn)
    t0 = kt[0] + pd.Timedelta(minutes=120)
    return meal_response(native_from_knots(kt, kv), t0, device)


def test_iauc_flat_curve_is_zero():
    r = _response_for(lambda k: 100.0)
    assert r["ok"] and np.isclose(r["iauc"], 0) and np.isclose(r["peak_rise"], 0)


def test_iauc_known_triangle():
    # baseline 100 before the meal (t0 = 120 min), linear rise to 160 at +60, back to 100 at +120
    def fn(k):
        rel = k - 120
        if rel <= 0 or rel >= 120:
            return 100.0
        return 100 + 60 * (1 - abs(rel - 60) / 60)
    r = _response_for(fn)
    assert r["ok"]
    assert np.isclose(r["iauc"], 0.5 * 120 * 60 / 60)  # mg/dL*h
    assert np.isclose(r["peak_rise"], 60)
    assert np.isclose(r["time_to_peak"], 60)


def test_iauc_ignores_dip_below_baseline():
    def fn(k):
        rel = k - 120
        if 0 < rel < 60:
            return 100 - 20 * (1 - abs(rel - 30) / 30)  # dip
        return 100.0
    r = _response_for(fn)
    assert r["ok"] and np.isclose(r["iauc"], 0)


def test_gap_in_window_excludes():
    ts, _, _ = make_minute_frame(minutes=400, native_period=5)
    t0 = ts.index[120]
    ts.iloc[150:190, ts.columns.get_loc("dexcom")] = np.nan
    p = make_participant(ts, [t0])
    r = meal_response(p.native["dexcom"], t0, "dexcom")
    assert not r["ok"] and r["reason"] == "cgm_gap_in_window"


def test_merge_and_overlap_rules():
    t = pd.to_datetime(["2020-01-01 08:00", "2020-01-01 08:20", "2020-01-01 08:45",
                        "2020-01-01 12:00", "2020-01-01 13:30", "2020-01-01 19:00"])
    m = pd.DataFrame({"time": t, "meal_type": ["breakfast", "snack", "snack", "lunch", "snack", "dinner"],
                      "kcal": 100.0, "carbs": 10.0, "protein": 5.0, "fat": 2.0, "fiber": 1.0,
                      "amount_consumed": 100.0, "has_photo": True})
    ep = merge_episodes(m)
    # 08:00 and 08:20 merge (within 30 min of first); 08:45 is 45 min after first -> new episode
    assert list(ep["n_records"]) == [2, 1, 1, 1, 1]
    assert ep.loc[0, "carbs"] == 20 and ep.loc[0, "meal_type"] == "breakfast"
    ep = flag_overlap(ep)
    # 08:00 is followed 45 min later; 12:00 is followed 90 min later -> both overlap
    assert list(ep["overlap_next"]) == [True, False, True, False, False]
    assert np.isclose(ep.loc[1, "min_since_prev"], 45)


def test_implausible_rules():
    ep = pd.DataFrame({"kcal": [400, 0, 100, 2000], "carbs": [50, 10, 5, 50],
                       "protein": [20, 1, 1, 10], "fat": [10, 1, 0, 10], "fiber": [5, 0, 8, 2]})
    assert list(implausible(ep)) == [False, True, True, True]


def test_build_meal_table_end_to_end():
    ts, _, _ = make_minute_frame(minutes=900, native_period=5)
    p = make_participant(ts, [ts.index[100], ts.index[400], ts.index[460]])
    ep, curves = meals.build_meal_table(p)
    ex = meals.exclusion_counts(ep, "dexcom")
    assert ex["episodes"] == 3 and ex["excl_overlap_next_meal"] == 1 and ex["eligible"] == 2
    assert curves["dexcom"].shape == (3, len(meals.CURVE_GRID))
