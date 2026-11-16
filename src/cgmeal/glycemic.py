"""Participant-level glycemic metrics and CGM-to-CGM agreement.

All metrics are computed on *native* sensor readings (see
:func:`cgmeal.io.native_readings`), not on the provider's 1-min interpolation.

Thresholds (Battelino et al., Diabetes Care 2019;42:1593-1603, international
consensus on time in range):
  TIR 70-180 mg/dL, TBR <70 and <54 mg/dL, TAR >180 and >250 mg/dL.
Time in 70-140 mg/dL is reported because TIR 70-180 saturates in people
without diabetes; Danne et al. 2017 (Table 1) list 70-140 mg/dL as a
secondary target range (no percentage target is attached to it here).
CV = 100 * SD / mean.  Danne et al., Diabetes Care 2017;40:1631-1640 (Section 5
and Table 1) define stable glucose as CV < 36 % and unstable as CV >= 36 %;
Battelino et al. 2019 (Table 2) give a target of CV <= 36 %.
MAGE (Service et al., Diabetes 1970;19:644-655): see :func:`mage`.

Percentages of time are computed as the fraction of native readings, which
equals the fraction of time because readings are equally spaced within a
device (gaps simply contribute no readings).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _minutes(t) -> np.ndarray:
    return pd.DatetimeIndex(t).values.astype("datetime64[s]").astype(np.int64) / 60.0


def gap_intervals(native: pd.DataFrame, max_gap_min: float, collinear_max_min: float | None = None) -> np.ndarray:
    """Boolean per interval (reading i -> i+1): True if it is treated as missing data.

    An interval longer than ``max_gap_min`` is a gap unless the reading carries
    ``regular_after`` (provider series continuous on a 1-min grid) and the
    interval is at most ``collinear_max_min`` long (collinear readings).
    """
    from .io import COLLINEAR_MAX_MIN

    cmax = COLLINEAR_MAX_MIN if collinear_max_min is None else collinear_max_min
    tn = _minutes(native["time"])
    d = np.diff(tn)
    reg = native["regular_after"].to_numpy(bool)[:-1] if "regular_after" in native else np.zeros(len(d), bool)
    return (d > max_gap_min) & ~(reg & (d <= cmax))


def interpolate_native(native: pd.DataFrame, query_times, max_gap_min: float) -> np.ndarray:
    """Linear interpolation of native readings at ``query_times``.

    Returns NaN outside the range of readings and inside any interval treated
    as missing data by :func:`gap_intervals`.
    """
    tn = _minutes(native["time"])
    gn = native["glucose"].to_numpy(float)
    tq = _minutes(query_times)
    out = np.interp(tq, tn, gn, left=np.nan, right=np.nan)
    if len(tn) < 2:
        return np.where(np.isin(tq, tn), out, np.nan)
    j = np.searchsorted(tn, tq, side="right")  # tn[j-1] <= tq < tn[j]
    j = np.clip(j, 1, len(tn) - 1)
    is_gap = gap_intervals(native, max_gap_min)[j - 1]
    exact = np.isin(tq, tn)
    out[is_gap & ~exact] = np.nan
    return out


def mage(glucose: np.ndarray, sd_multiplier: float = 1.0) -> float:
    """Mean amplitude of glycemic excursions (Service et al. 1970), simplified.

    Implementation used here (one of several published variants; Baghurst,
    Diabetes Technol Ther 2011;13:296-302 discusses why implementations differ):
      1. compute SD of the whole series;
      2. find turning points (local peaks and nadirs, plateaus collapsed);
      3. iteratively remove the smallest peak-nadir pair whose amplitude is
         below ``sd_multiplier`` * SD (merging neighbours) until every
         remaining consecutive amplitude is >= the threshold;
      4. MAGE = mean of the remaining consecutive amplitudes, counted in the
         direction (up or down) of the first qualifying excursion.
    Applied to one contiguous series (gaps are ignored, i.e. the series is
    concatenated); returns NaN if no excursion qualifies.
    """
    g = np.asarray(glucose, float)
    g = g[np.isfinite(g)]
    if len(g) < 3:
        return np.nan
    sd = np.std(g, ddof=1)
    thr = sd_multiplier * sd
    # collapse plateaus
    keep = np.r_[True, np.diff(g) != 0]
    x = g[keep]
    if len(x) < 3:
        return np.nan
    d = np.sign(np.diff(x))
    tp = [0] + [i for i in range(1, len(x) - 1) if d[i - 1] != d[i]] + [len(x) - 1]
    pts = list(x[tp])
    # iteratively remove small excursions
    changed = True
    while changed and len(pts) >= 3:
        changed = False
        amps = np.abs(np.diff(pts))
        k = int(np.argmin(amps))
        if amps[k] < thr:
            # remove the pair (k, k+1) unless at an end; then drop the end point
            if k == 0:
                del pts[0]
            elif k == len(pts) - 2:
                del pts[-1]
            else:
                a, b = pts[k], pts[k + 1]
                # pts[k-1] is the same type as b, pts[k+2] the same type as a;
                # keep the more extreme turning point of each type
                pick_prev = max if pts[k - 1] > a else min
                pick_next = max if pts[k + 2] > b else min
                pts[k - 1] = pick_prev(pts[k - 1], b)
                pts[k + 2] = pick_next(pts[k + 2], a)
                del pts[k:k + 2]
            changed = True
    diffs = np.diff(pts)
    diffs = diffs[np.abs(diffs) >= thr]
    if len(diffs) == 0:
        return np.nan
    direction = np.sign(diffs[0])
    return float(np.mean(np.abs(diffs[np.sign(diffs) == direction])))


def glycemic_metrics(glucose: np.ndarray) -> dict:
    """Summary metrics from native readings (mg/dL)."""
    g = np.asarray(glucose, float)
    g = g[np.isfinite(g)]
    n = len(g)
    if n == 0:
        return {}
    mean = float(np.mean(g))
    sd = float(np.std(g, ddof=1)) if n > 1 else np.nan
    return {
        "n_readings": n,
        "mean_glucose": mean,
        "sd_glucose": sd,
        "cv_pct": 100 * sd / mean,
        "cv_stable_lt36": float(100 * sd / mean < 36),
        "tir_70_180_pct": 100 * np.mean((g >= 70) & (g <= 180)),
        "titr_70_140_pct": 100 * np.mean((g >= 70) & (g <= 140)),
        "tar_180_pct": 100 * np.mean(g > 180),
        "tar_250_pct": 100 * np.mean(g > 250),
        "tbr_70_pct": 100 * np.mean(g < 70),
        "tbr_54_pct": 100 * np.mean(g < 54),
        "mage": mage(g),
    }


def paired_readings(native_ref: pd.DataFrame, native_other: pd.DataFrame,
                    max_gap_other: float) -> pd.DataFrame:
    """Pair each reference reading with the other device interpolated at the same time.

    Only readings recorded while both devices were worn (both defined) are kept.
    """
    other = interpolate_native(native_other, native_ref["time"], max_gap_other)
    df = pd.DataFrame({"time": native_ref["time"].to_numpy(),
                       "ref": native_ref["glucose"].to_numpy(float), "other": other})
    return df.dropna().reset_index(drop=True)


def agreement(pairs: pd.DataFrame, pid_col: str = "pid") -> dict:
    """Agreement between two CGMs (other minus reference).

    * bias: mean difference (mg/dL);
    * MARD: mean |other - ref| / ref * 100 (reference = Dexcom here; neither
      device is a laboratory reference, so this is a device-vs-device figure);
    * Bland-Altman 95 % limits of agreement (Bland & Altman, Lancet 1986).
      With repeated readings per participant, the SD of differences is taken
      as sqrt(between-participant variance of mean differences + mean
      within-participant variance), a simplified version of the
      variance-components approach of Bland & Altman, J Biopharm Stat
      2007;17:571-582.  Bias is the mean of participant means (each person
      weighted equally).
    """
    d = pairs["other"] - pairs["ref"]
    rel = (d.abs() / pairs["ref"]) * 100
    tmp = pairs.assign(d=d, rel=rel)
    per = tmp.groupby(pid_col).agg(bias=("d", "mean"), sd=("d", "std"), mard=("rel", "mean"), n=("d", "size"))
    bias = per["bias"].mean()
    sd_total = np.sqrt(per["bias"].var(ddof=1) + (per["sd"] ** 2).mean())
    return {
        "n_participants": int(len(per)),
        "n_pairs": int(len(tmp)),
        "bias_mgdl": float(bias),
        "loa_low": float(bias - 1.96 * sd_total),
        "loa_high": float(bias + 1.96 * sd_total),
        "mard_pct": float(per["mard"].mean()),
        "mard_pct_pooled": float(rel.mean()),
        "pearson_r": float(np.corrcoef(pairs["ref"], pairs["other"])[0, 1]),
        "per_participant_bias_median": float(per["bias"].median()),
        "per_participant_bias_q1": float(per["bias"].quantile(0.25)),
        "per_participant_bias_q3": float(per["bias"].quantile(0.75)),
        "per_participant_mard_median": float(per["mard"].median()),
    }


def lag_scan(natives_ref, natives_other, lags_min, max_gap_other: float, min_pairs: int = 100) -> pd.DataFrame:
    """Median within-participant correlation and bias when the other device is shifted.

    ``natives_ref`` / ``natives_other``: lists of native-reading DataFrames, one per
    participant.  The other device is evaluated at ref time + lag.
    """
    rows = []
    for lag in lags_min:
        rs, bs = [], []
        for nr, no in zip(natives_ref, natives_other):
            q = nr["time"] + pd.Timedelta(minutes=lag)
            v = interpolate_native(no, q, max_gap_other)
            ok = np.isfinite(v)
            if ok.sum() >= min_pairs:
                g = nr["glucose"].to_numpy(float)[ok]
                rs.append(np.corrcoef(g, v[ok])[0, 1])
                bs.append(np.mean(v[ok] - g))
        rows.append({"lag_min": lag, "median_r": float(np.median(rs)), "median_bias": float(np.median(bs)),
                     "n_participants": len(rs)})
    return pd.DataFrame(rows)
