import numpy as np
import pandas as pd

from cgmeal.glycemic import agreement, glycemic_metrics, interpolate_native, mage


def test_metrics_known_values():
    g = np.array([60, 100, 150, 200, 260] * 4, float)
    m = glycemic_metrics(g)
    assert np.isclose(m["mean_glucose"], 154)
    assert np.isclose(m["tir_70_180_pct"], 40)
    assert np.isclose(m["tar_180_pct"], 40)
    assert np.isclose(m["tar_250_pct"], 20)
    assert np.isclose(m["tbr_70_pct"], 20)
    assert np.isclose(m["cv_pct"], 100 * np.std(g, ddof=1) / g.mean())


def test_mage_square_wave():
    # alternating 100/200: every excursion is 100 mg/dL > 1 SD
    g = np.array([100, 200] * 10, float)
    assert np.isclose(mage(g), 100)


def test_mage_ignores_small_wiggles():
    base = np.array([100, 180] * 5, float)
    g = np.repeat(base, 6) + np.tile([0, 2, 0, 2, 0, 2], 10)
    assert abs(mage(g) - 80) <= 2


def test_interpolate_native_respects_gaps():
    t = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05", "2020-01-01 01:00"])
    nat = pd.DataFrame({"time": t, "glucose": [100.0, 110.0, 120.0]})
    q = pd.to_datetime(["2020-01-01 00:02", "2020-01-01 00:30", "2020-01-01 01:00", "2020-01-01 02:00"])
    v = interpolate_native(nat, q, max_gap_min=15)
    assert np.isclose(v[0], 104)
    assert np.isnan(v[1])  # inside a 55-min gap
    assert np.isclose(v[2], 120)  # exact reading
    assert np.isnan(v[3])  # beyond range


def test_agreement_constant_bias():
    rng = np.random.default_rng(0)
    ref = rng.uniform(80, 200, 400)
    pairs = pd.DataFrame({"pid": np.repeat([1, 2, 3, 4], 100), "ref": ref, "other": ref - 10})
    a = agreement(pairs)
    assert np.isclose(a["bias_mgdl"], -10)
    assert np.isclose(a["loa_low"], -10) and np.isclose(a["loa_high"], -10)
    assert np.isclose(a["mard_pct_pooled"], np.mean(10 / ref) * 100)
