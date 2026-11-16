"""Loading and aligning CGMacros v1.0.0.

Layout (as verified on the downloaded archive)
-----------------------------------------------
``data/CGMacros/bio.csv``                      one row per participant (demographics, labs)
``data/CGMacros/CGMacros-0XX/CGMacros-0XX.csv`` one row per minute for participant XX

Per-minute files
~~~~~~~~~~~~~~~~
* ``Timestamp`` -- date-shifted by the data providers (time of day is kept).
  Mostly a 1-min grid; some files contain jumps (2-, 3- or 901-min steps).
* ``Libre GL`` / ``Dexcom GL`` (mg/dL) -- **linearly interpolated** to 1 min
  from the native sensor readings (FreeStyle Libre Pro: 15 min; Dexcom G6
  Pro: 5 min).  A 1-min value between two native readings therefore depends
  on the *next* native reading, i.e. on the future.  :func:`native_readings`
  reconstructs the native readings as the knots of the piecewise-linear
  series (points where the second difference is non-zero, plus segment ends).
  If two consecutive native readings are exactly collinear with a third, the
  middle one is not recoverable; this loses nothing for linear interpolation
  and can only make "the latest available reading" *older*, never newer, so
  it cannot introduce look-ahead.
* ``HR`` (bpm, Fitbit Sense, mean over the last minute).
* ``Calories (Activity)`` (kcal per minute, Fitbit estimate).  Missing in one
  file (participant 7, which has ``Steps`` instead).
* ``METs`` (Fitbit METs x 10) in 34 files; 10 files have an ordinal Fitbit
  ``Intensity`` (0-3) instead.  We expose ``mets`` (divided by 10) and
  ``intensity`` and leave the other one missing.
* Meal rows: ``Meal Type`` (inconsistent case and spelling, normalised to
  breakfast/lunch/dinner/snack), ``Calories``, ``Carbs``, ``Protein``,
  ``Fat``, ``Fiber`` (g), ``Amount Consumed`` (%; absent in 2 files) and
  ``Image path``.  Rows with an image path but no meal type are the
  "after eating" photos.

Units: glucose mg/dL, macronutrients g, energy kcal, HbA1c % (the official
data dictionary says "mmol/mol" but the values 4.6-8.5 are clearly %).

Health-state group (as in the dataset description, based on screening HbA1c):
healthy < 5.7 %, prediabetes 5.7-6.4 %, type 2 diabetes > 6.4 %.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ["healthy", "prediabetes", "t2d"]
DEVICES = {"dexcom": "Dexcom GL", "libre": "Libre GL"}
NATIVE_PERIOD_MIN = {"dexcom": 5, "libre": 15}
# Longest interval between consecutive recovered readings that is always
# interpolated across (three native periods).
MAX_GAP_MIN = {"dexcom": 15, "libre": 45}
# A longer interval is still accepted as *collinear readings* (readings whose
# knot is invisible because they lie on a straight line, e.g. a flat stretch)
# when the provider series between the two readings sits on a regular 1-min
# grid with no missing value, up to this length.  Missed readings appear in
# the files as missing values (Dexcom) or as irregular 2-7 min / 901 min
# timestamp steps (Libre; see Saha 2026), and those intervals stay gaps.
COLLINEAR_MAX_MIN = 60

BIO_COLUMNS = {
    "subject": "pid",
    "Age": "age",
    "Gender": "sex",
    "BMI": "bmi",
    "A1c PDL (Lab)": "hba1c",
    "Fasting GLU - PDL (Lab)": "fasting_glucose",
    "Insulin": "insulin",
    "Triglycerides": "triglycerides",
    "Cholesterol": "cholesterol",
    "HDL": "hdl",
    "Non HDL": "non_hdl",
    "LDL (Cal)": "ldl",
    "VLDL (Cal)": "vldl",
}


def hba1c_group(hba1c: float) -> str:
    """Health-state group from HbA1c (%), using the CGMacros definitions."""
    if pd.isna(hba1c):
        return np.nan
    if hba1c < 5.7:
        return "healthy"
    if hba1c <= 6.4:
        return "prediabetes"
    return "t2d"


def load_participants(data_dir: str | Path = "data") -> pd.DataFrame:
    """Participant-level table: demographics, labs and health-state group."""
    bio = pd.read_csv(Path(data_dir) / "CGMacros" / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    df = bio[list(BIO_COLUMNS)].rename(columns=BIO_COLUMNS)
    df["pid"] = df["pid"].astype(int)
    # Error codes documented in the data dictionary
    df.loc[df["ldl"] >= 800, "ldl"] = np.nan
    df.loc[df["vldl"] >= 400, "vldl"] = np.nan
    df["sex_male"] = (df["sex"].str.strip().str.upper() == "M").astype(float)
    df["group"] = df["hba1c"].map(hba1c_group)
    return df.set_index("pid").sort_index()


def _norm_meal_type(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z]", "", s)
    if s.startswith("snack"):
        return "snack"
    return s


@dataclass
class Participant:
    pid: int
    ts: pd.DataFrame  # 1-min signals indexed by Timestamp
    meals: pd.DataFrame  # one row per logged meal record
    native: dict = field(default_factory=dict)  # device -> DataFrame(time, glucose)


def read_minute_file(path: str | Path) -> pd.DataFrame:
    """Read one per-minute file into harmonised columns, sorted by time."""
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d["Timestamp"] = pd.to_datetime(d["Timestamp"])
    d = d.sort_values("Timestamp").drop_duplicates("Timestamp", keep="first")
    out = pd.DataFrame(index=pd.DatetimeIndex(d["Timestamp"].values, name="time"))
    out["libre"] = d["Libre GL"].to_numpy(float)
    out["dexcom"] = d["Dexcom GL"].to_numpy(float)
    out["hr"] = d["HR"].to_numpy(float)
    out["activity_kcal"] = d["Calories (Activity)"].to_numpy(float) if "Calories (Activity)" in d else np.nan
    out["mets"] = d["METs"].to_numpy(float) / 10.0 if "METs" in d else np.nan
    out["intensity"] = d["Intensity"].to_numpy(float) if "Intensity" in d else np.nan
    out["meal_type"] = d["Meal Type"].to_numpy(object)
    for src, dst in [("Calories", "kcal"), ("Carbs", "carbs"), ("Protein", "protein"),
                     ("Fat", "fat"), ("Fiber", "fiber"), ("Amount Consumed", "amount_consumed")]:
        out[dst] = d[src].to_numpy(float) if src in d else np.nan
    out["image"] = d["Image path"].to_numpy(object)
    return out


def extract_meals(raw: pd.DataFrame) -> pd.DataFrame:
    """Meal records (rows with a meal type), harmonised."""
    m = raw[raw["meal_type"].notna()].copy()
    m["meal_type"] = m["meal_type"].map(_norm_meal_type)
    m["has_photo"] = m["image"].notna()
    m = m.reset_index().rename(columns={"index": "time"})
    cols = ["time", "meal_type", "kcal", "carbs", "protein", "fat", "fiber", "amount_consumed", "has_photo"]
    return m[cols].reset_index(drop=True)


def native_readings(series: pd.Series, tol: float = 1e-6, max_step_min: int = 5) -> pd.DataFrame:
    """Reconstruct native sensor readings from a 1-min linearly interpolated series.

    The series is split into segments of non-missing rows whose timestamps are at
    most ``max_step_min`` apart (missing values or longer timestamp jumps start a
    new segment).  Within a segment the native readings are the knots of the
    piecewise-linear curve: both segment ends and every interior point where the
    slope (computed with the actual time steps) changes by more than ``tol``
    mg/dL/min.  Returns a DataFrame with columns ``time``, ``glucose`` and
    ``regular_after`` (True if the rows between this reading and the next one
    are consecutive 1-min rows without missing values, i.e. an interval that
    can only hide collinear readings, not missed ones).
    """
    s = series[~series.index.duplicated()].sort_index()
    finite = s.notna().to_numpy()
    t_all = s.index.values.astype("datetime64[m]").astype(np.int64)
    if not finite.any():
        return pd.DataFrame({"time": pd.to_datetime([]), "glucose": [], "regular_after": []})
    # a missing value between two present values must break the segment
    run_id = np.cumsum(np.r_[True, finite[1:] != finite[:-1]])
    t = t_all[finite]
    v = s.to_numpy(float)[finite]
    rid = run_id[finite]
    new_seg = np.r_[True, (np.diff(t) > max_step_min) | (np.diff(rid) != 0)]
    seg_id = np.cumsum(new_seg)
    keep = np.zeros(len(v), bool)
    for sid in np.unique(seg_id):
        idx = np.where(seg_id == sid)[0]
        keep[idx[0]] = keep[idx[-1]] = True
        if len(idx) >= 3:
            slope = np.diff(v[idx]) / np.diff(t[idx])
            keep[idx[1:-1][np.abs(np.diff(slope)) > tol]] = True
    times = s.index[finite][keep]
    # regular_after: same segment as the next kept reading and only 1-min steps between them
    irregular = np.r_[0, np.cumsum(np.diff(t) != 1)]  # count of non-1-min steps up to each row
    k = np.where(keep)[0]
    reg = np.zeros(len(k), bool)
    if len(k) > 1:
        reg[:-1] = (seg_id[k[:-1]] == seg_id[k[1:]]) & (irregular[k[1:]] == irregular[k[:-1]])
    return pd.DataFrame({"time": times, "glucose": v[keep], "regular_after": reg}).reset_index(drop=True)


def load_participant(data_dir: str | Path, pid: int) -> Participant:
    path = Path(data_dir) / "CGMacros" / f"CGMacros-{pid:03d}" / f"CGMacros-{pid:03d}.csv"
    raw = read_minute_file(path)
    meals = extract_meals(raw)
    ts = raw[["libre", "dexcom", "hr", "activity_kcal", "mets", "intensity"]]
    native = {dev: native_readings(ts[dev]) for dev in DEVICES}
    return Participant(pid=pid, ts=ts, meals=meals, native=native)


def load_all(data_dir: str | Path = "data") -> tuple[pd.DataFrame, dict[int, Participant]]:
    """Participant table and a dict pid -> Participant for all participants."""
    people = load_participants(data_dir)
    parts = {pid: load_participant(data_dir, pid) for pid in people.index}
    return people, parts


def count_photos(data_dir: str | Path = "data") -> int:
    """Number of photo files in the downloaded archive (photos are not extracted)."""
    import zipfile

    z = Path(data_dir) / "raw" / "CGMacros_dateshifted365.zip"
    if not z.exists():
        return -1
    with zipfile.ZipFile(z) as f:
        return sum(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in f.namelist())
