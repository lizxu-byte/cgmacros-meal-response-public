import numpy as np
import pandas as pd

from cgmeal.predict import loso_splits, personal_history, temporal_split, run_loso, metrics


def _df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, 7):
        n = rng.integers(8, 15)
        times = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.permutation(np.arange(n)) * 300, unit="min")
        for t in times:
            rows.append({"pid": pid, "time": t, "x": rng.normal(), "y": rng.normal()})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def test_loso_never_mixes_participants():
    df = _df()
    seen = []
    for tr, te in loso_splits(df["pid"]):
        test_ids = set(df.loc[te, "pid"])
        assert len(test_ids) == 1
        assert test_ids.isdisjoint(set(df.loc[tr, "pid"]))
        seen.extend(te)
    assert sorted(seen) == list(range(len(df)))


def test_temporal_split_ordering():
    df = _df(1)
    train = temporal_split(df, frac=0.7)
    for pid, sub in df.groupby("pid"):
        m = train[sub.index]
        assert m.any() and (~m).any()
        assert sub.loc[m, "time"].max() < sub.loc[~m, "time"].min()
        assert m.sum() == int(np.floor(0.7 * len(sub)))


def test_personal_history_uses_training_rows_only():
    y = np.array([1.0, 2.0, 3.0, 100.0, 10.0, 20.0, 999.0])
    g = np.array([1, 1, 1, 1, 2, 2, 2])
    train = np.array([True, True, True, False, True, True, False])
    h = personal_history(y, g, train)
    assert np.allclose(h[:3], [2.5, 2.0, 1.5])  # leave-one-out within training
    assert np.isclose(h[3], 2.0)  # test row: mean of training rows, never its own y
    assert np.isclose(h[6], 15.0)


def test_run_loso_predicts_every_row():
    df = _df(2)
    df["y"] = 2 * df["x"] + 0.1 * np.random.default_rng(0).normal(size=len(df))
    yhat = run_loso(df[["x"]], df["y"], df["pid"], "ridge")
    assert np.isfinite(yhat).all()
    m = metrics(df["y"], yhat, df["pid"])
    assert m["pooled_r"] > 0.95
