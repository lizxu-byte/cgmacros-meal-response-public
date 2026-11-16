import numpy as np
import pandas as pd

from cgmeal.models import fit_mixed, group_carb_slopes
from cgmeal.stats import cliffs_delta, compare_groups, holm


def test_holm_matches_hand_calculation():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_cliffs_delta_extremes():
    assert cliffs_delta([1, 2, 3], [4, 5, 6]) == 1
    assert cliffs_delta([4, 5, 6], [1, 2, 3]) == -1
    assert cliffs_delta([1, 2], [1, 2]) == 0


def test_compare_groups_detects_shift():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"group": np.repeat(["healthy", "prediabetes", "t2d"], 15),
                       "x": np.r_[rng.normal(0, 1, 15), rng.normal(1, 1, 15), rng.normal(3, 1, 15)]})
    t = compare_groups(df, ["x"]).iloc[0]
    assert t["kw_p"] < 0.001 and t["eta2_H"] > 0.5 and t["delta_healthy_vs_t2d"] > 0.9


def test_mixed_model_recovers_carb_by_group_interaction():
    rng = np.random.default_rng(1)
    rows = []
    slope = {"healthy": 1.0, "prediabetes": 1.5, "t2d": 2.0}  # sqrt-scale change per 10 g
    for pid in range(30):
        g = ["healthy", "prediabetes", "t2d"][pid % 3]
        u = rng.normal(0, 0.5)
        for _ in range(25):
            c = rng.uniform(10, 100)
            z = 12 + u + slope[g] * (c - 55) / 10 + rng.normal(0, 0.5)
            rows.append({"pid": pid, "group": g, "carbs": c, "protein": rng.uniform(5, 40),
                         "fat": rng.uniform(5, 40), "fiber": rng.uniform(0, 10), "hr_pre": rng.normal(75, 8),
                         "activity_kcal_pre": rng.uniform(1, 3), "hour_sin": rng.uniform(-1, 1),
                         "hour_cos": rng.uniform(-1, 1), "log_min_since_prev": rng.uniform(4, 6.5),
                         "pre_glucose": rng.normal(100, 10), "iauc": max(z, 0) ** 2})
    fit = fit_mixed(pd.DataFrame(rows), "iauc", transform="sqrt")
    s = group_carb_slopes(fit).set_index("group")["slope_per10g"]
    assert abs(s["healthy"] - 1.0) < 0.15 and abs(s["t2d"] - 2.0) < 0.15
