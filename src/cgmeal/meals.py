"""Meal episodes and postprandial glucose responses.

Rules (all parameters are module constants so they are visible in one place)
--------------------------------------------------------------------------
1. **Episodes.** Meal records are grouped into an episode when a record starts
   within ``MERGE_WINDOW_MIN`` (30 min) of the *first* record of the episode;
   nutrients are summed, the episode time is the first record's time and the
   type is the first record's type.
2. **Implausible nutrient entries** are excluded: energy == 0 kcal, fibre >
   carbohydrate + 1 g, or reported energy differing from the Atwater estimate
   (4 C + 4 P + 9 F kcal) by more than a factor of 2.
3. **Overlap.** An episode is excluded from response analyses when another
   episode starts less than ``WINDOW_MIN`` (120 min) after it (its 2-h
   response is contaminated).  Earlier meals are not an exclusion reason; the
   time since the previous episode is kept as a predictor.
4. **Missing CGM.** For each device, a response is computed only if
   (a) at least one native reading lies in the baseline window
   [-``BASELINE_MIN``, 0] min, and (b) the native readings cover [0, 120] min
   with no gap.  An interval longer than the device's ``MAX_GAP_MIN`` is a gap
   only if the provider series has a missing value or an irregular timestamp
   step inside it, or it is longer than ``COLLINEAR_MAX_MIN`` (60 min);
   otherwise it holds collinear readings (flat or constant-slope stretches)
   and is used.  Each response records how many such long intervals it
   contained (``n_long_flat``, ``n_long_sloped``) so the effect of this rule
   is reported.

Response definitions
--------------------
* baseline: mean of native readings in [-30, 0] min (past data only);
* iAUC (mg/dL x h) over 0-120 min: area of the glucose curve above baseline,
  with area below baseline ignored (the "incremental area" of the glycaemic
  index method; Brouns et al., Nutr Res Rev 2005;18:145-171). The glucose
  curve is the linear interpolation between native readings; the positive
  area is integrated exactly, including the triangles at baseline crossings;
* peak rise: max(glucose over 0-120 min) - baseline (mg/dL);
* time to peak: minutes from meal start to that maximum.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .glycemic import gap_intervals, interpolate_native
from .io import MAX_GAP_MIN

MERGE_WINDOW_MIN = 30
WINDOW_MIN = 120
BASELINE_MIN = 30
CURVE_GRID = np.arange(-30, 181, 5)  # minutes, for average curves

NUTRIENTS = ["kcal", "carbs", "protein", "fat", "fiber"]


def merge_episodes(meals: pd.DataFrame, window_min: float = MERGE_WINDOW_MIN) -> pd.DataFrame:
    """Group meal records into eating episodes (rule 1)."""
    m = meals.sort_values("time").reset_index(drop=True)
    if m.empty:
        return m.assign(n_records=[])
    ep = np.zeros(len(m), int)
    start = m["time"].iloc[0]
    k = 0
    for i, t in enumerate(m["time"]):
        if (t - start).total_seconds() / 60 > window_min:
            k += 1
            start = t
        ep[i] = k
    m["episode"] = ep
    agg = {c: "sum" for c in NUTRIENTS}
    agg.update(time="first", meal_type="first", has_photo="any", amount_consumed="first")
    out = m.groupby("episode").agg(agg)
    out["n_records"] = m.groupby("episode").size()
    # sum() of all-NaN gives 0; keep NaN if every record was missing
    for c in NUTRIENTS:
        out.loc[m.groupby("episode")[c].apply(lambda s: s.isna().all()), c] = np.nan
    return out.reset_index(drop=True)


def implausible(ep: pd.DataFrame) -> pd.Series:
    """Rule 2: implausible nutrient entries."""
    atwater = 4 * ep["carbs"] + 4 * ep["protein"] + 9 * ep["fat"]
    ratio = ep["kcal"] / atwater.replace(0, np.nan)
    bad = (ep["kcal"] <= 0) | (ep["fiber"] > ep["carbs"] + 1) | (ratio > 2) | (ratio < 0.5)
    return bad.fillna(True)


def flag_overlap(ep: pd.DataFrame, window_min: float = WINDOW_MIN) -> pd.DataFrame:
    """Rule 3 plus time since previous episode (minutes; NaN for the first)."""
    ep = ep.sort_values("time").reset_index(drop=True)
    t = ep["time"]
    nxt = (t.shift(-1) - t).dt.total_seconds() / 60
    prev = (t - t.shift(1)).dt.total_seconds() / 60
    ep["min_to_next"] = nxt
    ep["min_since_prev"] = prev
    ep["overlap_next"] = nxt < window_min
    return ep


def positive_area(t: np.ndarray, y: np.ndarray) -> float:
    """Exact area of max(y, 0) for the piecewise-linear curve through (t, y)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    area = 0.0
    for t0, t1, y0, y1 in zip(t[:-1], t[1:], y[:-1], y[1:]):
        dt = t1 - t0
        if y0 >= 0 and y1 >= 0:
            area += dt * (y0 + y1) / 2
        elif y0 > 0 or y1 > 0:  # one positive, one negative: triangle
            pos, neg = (y0, -y1) if y0 > 0 else (y1, -y0)
            area += dt * pos * pos / (2 * (pos + neg))
    return area


def _rel_minutes(times, t0) -> np.ndarray:
    return (pd.DatetimeIndex(times) - t0).total_seconds().to_numpy() / 60.0


def meal_response(native: pd.DataFrame, t0: pd.Timestamp, device: str,
                  window_min: float = WINDOW_MIN, baseline_min: float = BASELINE_MIN) -> dict:
    """Postprandial response for one meal from one device's native readings.

    Returns a dict with ``ok`` (bool), ``reason`` (exclusion reason or ""), and
    when ok: baseline, iauc (mg/dL*h), peak_rise, time_to_peak, plus the curve
    of glucose minus baseline on :data:`CURVE_GRID` (NaN where not covered).
    """
    max_gap = MAX_GAP_MIN[device]
    rel = _rel_minutes(native["time"], t0)
    g = native["glucose"].to_numpy(float)
    base_mask = (rel >= -baseline_min) & (rel <= 0)
    if not base_mask.any():
        return {"ok": False, "reason": "no_baseline_reading"}
    baseline = float(g[base_mask].mean())
    # native readings bracketing [0, window]
    i0 = np.searchsorted(rel, 0, side="right") - 1  # last reading <= 0
    i1 = np.searchsorted(rel, window_min, side="left")  # first reading >= window
    if i0 < 0 or i1 >= len(rel):
        return {"ok": False, "reason": "cgm_not_covering_window"}
    seg_t, seg_g = rel[i0:i1 + 1], g[i0:i1 + 1]
    long_iv = np.diff(seg_t) > max_gap
    if np.any(gap_intervals(native, max_gap)[i0:i1]):
        return {"ok": False, "reason": "cgm_gap_in_window"}
    flat = long_iv & (np.diff(seg_g) == 0)
    counts = {"n_long_flat": int(flat.sum()), "n_long_sloped": int((long_iv & ~flat).sum())}
    # curve on [0, window] using readings inside plus interpolated end points
    grid = np.unique(np.r_[0.0, seg_t[(seg_t > 0) & (seg_t < window_min)], window_min])
    yg = np.interp(grid, seg_t, seg_g)
    y = yg - baseline
    iauc = positive_area(grid, y) / 60.0
    k = int(np.argmax(yg))
    curve_t = t0 + pd.to_timedelta(CURVE_GRID, unit="min")
    curve = interpolate_native(native, curve_t, max_gap) - baseline
    return {"ok": True, "reason": "", "baseline": baseline, "iauc": iauc,
            "peak_rise": float(yg[k] - baseline), "time_to_peak": float(grid[k]),
            "curve": curve, **counts}


def build_meal_table(part, devices=("dexcom", "libre")) -> tuple[pd.DataFrame, dict]:
    """Episodes for one participant with responses for each device.

    Returns (table, curves) where curves[device] is an array (n_episodes x grid).
    The table holds one row per episode with exclusion flags; it is used in
    memory only and never written to ``results/``.
    """
    ep = flag_overlap(merge_episodes(part.meals))
    ep["implausible"] = implausible(ep)
    ep["pid"] = part.pid
    curves = {}
    for dev in devices:
        rows, cv = [], []
        for t0 in ep["time"]:
            r = meal_response(part.native[dev], t0, dev)
            cv.append(r.pop("curve", np.full(len(CURVE_GRID), np.nan)))
            rows.append(r)
        r = pd.DataFrame(rows).add_prefix(f"{dev}_")
        ep = pd.concat([ep, r], axis=1)
        curves[dev] = np.vstack(cv) if cv else np.empty((0, len(CURVE_GRID)))
    return ep, curves


def eligible(ep: pd.DataFrame, device: str) -> pd.Series:
    """Episodes usable for response analyses for a device."""
    return (~ep["implausible"]) & (~ep["overlap_next"]) & ep[f"{device}_ok"].fillna(False).astype(bool)


def exclusion_counts(ep: pd.DataFrame, device: str) -> dict:
    """Sequential exclusion counts (each episode counted at its first failing rule)."""
    n = len(ep)
    bad = ep["implausible"]
    ov = ~bad & ep["overlap_next"]
    rest = ~bad & ~ep["overlap_next"]
    reason = ep[f"{device}_reason"].where(rest)
    out = {"episodes": n, "excl_implausible_nutrients": int(bad.sum()), "excl_overlap_next_meal": int(ov.sum())}
    for r in ["no_baseline_reading", "cgm_not_covering_window", "cgm_gap_in_window"]:
        out[f"excl_{r}"] = int((reason == r).sum())
    out["eligible"] = int(eligible(ep, device).sum())
    ok = eligible(ep, device)
    flat = ep.get(f"{device}_n_long_flat", pd.Series(0, index=ep.index)).fillna(0)
    sloped = ep.get(f"{device}_n_long_sloped", pd.Series(0, index=ep.index)).fillna(0)
    out["eligible_with_long_flat_only"] = int((ok & (flat > 0) & (sloped == 0)).sum())
    out["eligible_with_long_sloped_only"] = int((ok & (sloped > 0) & (flat == 0)).sum())
    out["eligible_with_long_flat_and_sloped"] = int((ok & (flat > 0) & (sloped > 0)).sum())
    return out
