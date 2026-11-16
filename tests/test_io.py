import numpy as np
import pandas as pd

from cgmeal.io import hba1c_group, native_readings

from conftest import make_minute_frame


def test_native_readings_recovers_knots():
    ts, kt, kv = make_minute_frame(minutes=300, native_period=15, seed=3)
    nat = native_readings(ts["libre"])
    # every recovered point is a true knot
    assert set(nat["time"]).issubset(set(kt))
    # interpolating the recovered knots reproduces the series exactly
    rec = np.interp(ts.index.asi8, nat["time"].astype("int64"), nat["glucose"])
    ok = ts["libre"].notna()
    assert np.allclose(rec[ok], ts["libre"][ok])


def test_native_readings_breaks_on_missing():
    ts, _, _ = make_minute_frame(minutes=120, native_period=5)
    ts.iloc[40:60, ts.columns.get_loc("dexcom")] = np.nan
    nat = native_readings(ts["dexcom"])
    t = nat["time"]
    assert not ((t > ts.index[39]) & (t < ts.index[60])).any()
    # last value before the gap and first after it are kept as segment ends
    assert ts.index[39] in set(t) and ts.index[60] in set(t)


def test_hba1c_group_thresholds():
    assert hba1c_group(5.6) == "healthy"
    assert hba1c_group(5.7) == "prediabetes"
    assert hba1c_group(6.4) == "prediabetes"
    assert hba1c_group(6.5) == "t2d"
    assert pd.isna(hba1c_group(np.nan))
