"""Group comparisons for small samples (15 / 16 / 14 participants).

Unit of analysis is the participant: meal-level responses are first reduced to
one value per participant (median), so meals from the same person are not
treated as independent.  Tests are rank-based because groups are small and
the metrics are skewed:

* omnibus: Kruskal-Wallis H; effect size eta^2_H = (H - k + 1) / (n - k);
* pairwise: two-sided Mann-Whitney U with Holm correction over the three pairs;
  effect size Cliff's delta = P(X > Y) - P(X < Y) (positive = second group
  higher);
* across metrics, Kruskal-Wallis p-values are additionally Holm-adjusted.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def cliffs_delta(x, y) -> float:
    """P(Y > X) - P(Y < X) for samples x (reference) and y."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    gt = (y[:, None] > x[None, :]).sum()
    lt = (y[:, None] < x[None, :]).sum()
    return float((gt - lt) / (len(x) * len(y)))


def compare_groups(df: pd.DataFrame, metrics: list[str], group_col: str = "group",
                   order=("healthy", "prediabetes", "t2d")) -> pd.DataFrame:
    """One row per metric: medians/IQR by group, Kruskal-Wallis and pairwise tests."""
    rows = []
    for m in metrics:
        samples = [df.loc[df[group_col] == g, m].dropna().to_numpy(float) for g in order]
        r = {"metric": m}
        for g, s in zip(order, samples):
            r[f"{g}_n"] = len(s)
            r[f"{g}_median"] = np.median(s) if len(s) else np.nan
            r[f"{g}_q1"] = np.quantile(s, 0.25) if len(s) else np.nan
            r[f"{g}_q3"] = np.quantile(s, 0.75) if len(s) else np.nan
        valid = [s for s in samples if len(s) > 0]
        n = sum(len(s) for s in valid)
        k = len(valid)
        if k >= 2 and len(np.unique(np.concatenate(valid))) > 1:
            h, p = stats.kruskal(*valid)
            r.update(kw_H=h, kw_p=p, eta2_H=(h - k + 1) / (n - k))
        else:
            r.update(kw_H=np.nan, kw_p=np.nan, eta2_H=np.nan)
        pair_p = []
        for (i, a), (j, b) in combinations(enumerate(order), 2):
            x, y = samples[i], samples[j]
            if len(x) and len(y):
                p = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                r[f"delta_{a}_vs_{b}"] = cliffs_delta(x, y)
            else:
                p = np.nan
                r[f"delta_{a}_vs_{b}"] = np.nan
            pair_p.append((f"p_holm_{a}_vs_{b}", p))
        for (name, _), adj in zip(pair_p, holm([p for _, p in pair_p])):
            r[name] = adj
        rows.append(r)
    out = pd.DataFrame(rows)
    out["kw_p_holm_across_metrics"] = holm(out["kw_p"])
    return out
