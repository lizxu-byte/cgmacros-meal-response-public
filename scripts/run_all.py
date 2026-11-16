"""Run the full analysis on CGMacros and write aggregated tables/figures to results/.

Usage:  python scripts/run_all.py [--data-dir data] [--out results] [--n-perm 200]

Nothing at the meal or per-participant level is written: every table is
aggregated by health-state group or over the whole cohort.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from cgmeal import classify, features, glycemic, io, meals, models, plots, predict, stats

PRIMARY = "dexcom"
DEVICE_LABEL = {"dexcom": "Dexcom G6 Pro (5 min)", "libre": "FreeStyle Libre Pro (15 min)"}
CARB_BINS = [(15, 35), (35, 60), (60, 100)]
TARGETS = {"iauc": "2-h iAUC (mg/dL*h)", "peak_rise": "Peak rise (mg/dL)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_table(df: pd.DataFrame, path: Path, digits: int = 4):
    df.to_csv(path, index=True if df.index.name or not isinstance(df.index, pd.RangeIndex) else False,
              float_format=f"%.{digits}g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    # ------------------------------------------------------------------ load
    log("loading data")
    people, parts = io.load_all(args.data_dir)
    summary["n_participants"] = int(len(people))
    summary["group_n"] = people["group"].value_counts().reindex(io.GROUPS).astype(int).to_dict()
    cohort = people.groupby("group").agg(
        n=("age", "size"), age_median=("age", "median"), bmi_median=("bmi", "median"),
        hba1c_median=("hba1c", "median"), fasting_glucose_median=("fasting_glucose", "median"),
        insulin_median=("insulin", "median"), male_pct=("sex_male", lambda s: 100 * s.mean()),
    ).reindex(io.GROUPS)
    fmt_table(cohort, out / "table_cohort_by_group.csv")

    # data-coverage facts
    cov = {}
    for dev in io.DEVICES:
        cov[f"{dev}_native_readings"] = int(sum(len(p.native[dev]) for p in parts.values()))
    cov["participants_with_mets"] = int(sum(p.ts["mets"].notna().any() for p in parts.values()))
    cov["participants_with_intensity_only"] = int(sum(p.ts["intensity"].notna().any() for p in parts.values()))
    cov["participants_with_activity_kcal"] = int(sum(p.ts["activity_kcal"].notna().any() for p in parts.values()))
    cov["hr_missing_pct_median"] = float(np.median([100 * p.ts["hr"].isna().mean() for p in parts.values()]))
    cov["recording_days_median"] = float(np.median([(p.ts.index[-1] - p.ts.index[0]).total_seconds() / 86400
                                                     for p in parts.values()]))
    summary["coverage"] = cov

    # ------------------------------------------------ CGM agreement (validity)
    log("CGM agreement")
    pairs = []
    for pid, p in parts.items():
        pr = glycemic.paired_readings(p.native["dexcom"], p.native["libre"], io.MAX_GAP_MIN["libre"])
        pr["pid"] = pid
        pr["group"] = people.loc[pid, "group"]
        pairs.append(pr)
    pairs = pd.concat(pairs, ignore_index=True)
    agr = {"all": glycemic.agreement(pairs)}
    for g in io.GROUPS:
        agr[g] = glycemic.agreement(pairs[pairs["group"] == g])
    agr_df = pd.DataFrame(agr).T
    agr_df.index.name = "subset"
    fmt_table(agr_df, out / "table_cgm_agreement.csv")
    summary["cgm_agreement"] = agr["all"]
    lag = glycemic.lag_scan([p.native["dexcom"] for p in parts.values()], [p.native["libre"] for p in parts.values()],
                            range(-20, 21, 5), io.MAX_GAP_MIN["libre"])
    lag.to_csv(out / "table_cgm_lag_scan.csv", index=False, float_format="%.4g")
    summary["cgm_lag_scan"] = lag.to_dict("records")
    plots.bland_altman((pairs["ref"] + pairs["other"]) / 2, pairs["other"] - pairs["ref"], agr["all"],
                       out / "figures" / "fig1_bland_altman_libre_vs_dexcom.png")

    # ------------------------------------------- participant glycemic metrics
    log("glycemic metrics")
    glyc = {dev: pd.DataFrame({pid: glycemic.glycemic_metrics(p.native[dev]["glucose"])
                               for pid, p in parts.items()}).T.join(people["group"]) for dev in io.DEVICES}
    gm_cols = ["mean_glucose", "sd_glucose", "cv_pct", "tir_70_180_pct", "titr_70_140_pct", "tar_180_pct",
               "tar_250_pct", "tbr_70_pct", "tbr_54_pct", "mage"]
    for dev in io.DEVICES:
        fmt_table(stats.compare_groups(glyc[dev], gm_cols), out / f"table_glycemic_by_group_{dev}.csv")
    # CV stability threshold (Danne et al. 2017: stable < 36 %, unstable >= 36 %)
    cv_rows = []
    for dev in io.DEVICES:
        for g in io.GROUPS:
            sub = glyc[dev][glyc[dev]["group"] == g]
            cv_rows.append({"device": dev, "group": g, "n": len(sub),
                            "n_cv_lt36_stable": int(sub["cv_stable_lt36"].sum()),
                            "cv_max": float(sub["cv_pct"].max())})
    pd.DataFrame(cv_rows).to_csv(out / "table_cv_threshold_by_group.csv", index=False, float_format="%.4g")
    summary["cv_threshold"] = cv_rows

    # ---------------------------------------------------------------- meals
    log("meal responses and features")
    eps, curves, feats = [], {d: [] for d in io.DEVICES}, []
    for pid, p in parts.items():
        ep, cv = meals.build_meal_table(p)
        ep["group"] = people.loc[pid, "group"]
        eps.append(ep)
        for d in io.DEVICES:
            curves[d].append(cv[d])
        feats.append(features.build_features(p, ep, people, device=PRIMARY))
    ep = pd.concat(eps, ignore_index=True)
    curves = {d: np.vstack(v) for d, v in curves.items()}
    F = pd.concat(feats, ignore_index=True)

    # exclusions
    exc = {}
    for d in io.DEVICES:
        exc[d] = meals.exclusion_counts(ep, d)
        for g in io.GROUPS:
            exc[f"{d}_{g}"] = meals.exclusion_counts(ep[ep["group"] == g], d)
    exc_df = pd.DataFrame(exc).T
    exc_df.index.name = "device_subset"
    exc_df.insert(0, "meal_records", [int(sum(len(p.meals) for p in parts.values()))] * len(exc_df))
    fmt_table(exc_df, out / "table_meal_exclusions.csv")

    # 120-min overlap rule by group: pooled share and per-participant share
    ov_rows = []
    for g in io.GROUPS:
        e = ep[ep["group"] == g]
        plaus = e[~e["implausible"]]
        per = plaus.groupby("pid")["overlap_next"].mean() * 100
        ov_rows.append({"group": g, "episodes": len(e), "plausible_episodes": len(plaus),
                        "excluded_overlap": int(plaus["overlap_next"].sum()),
                        "pct_of_all_episodes": 100 * plaus["overlap_next"].sum() / len(e),
                        "pct_of_plausible_episodes": 100 * plaus["overlap_next"].mean(),
                        "per_participant_pct_median": per.median(), "per_participant_pct_q1": per.quantile(0.25),
                        "per_participant_pct_q3": per.quantile(0.75)})
    ov = pd.DataFrame(ov_rows)
    per_all = (ep[~ep["implausible"]].groupby("pid")["overlap_next"].mean() * 100).to_frame("pct").join(people["group"])
    kw = stats.compare_groups(per_all, ["pct"]).iloc[0]
    ov["kw_p_per_participant"] = kw["kw_p"]
    ov.to_csv(out / "table_overlap_exclusion_by_group.csv", index=False, float_format="%.4g")
    summary["overlap_by_group"] = ov.to_dict("records")
    summary["meal_records"] = int(sum(len(p.meals) for p in parts.values()))
    summary["exclusions"] = {d: exc[d] for d in io.DEVICES}
    summary["meal_type_counts_episodes"] = ep["meal_type"].value_counts().to_dict()

    # photos (counted, not modelled)
    rec = pd.concat([p.meals for p in parts.values()])
    photos = {"photo_files_in_archive": io.count_photos(args.data_dir),
              "meal_records": int(len(rec)), "meal_records_with_photo": int(rec["has_photo"].sum()),
              "episodes": int(len(ep)), "episodes_with_photo": int(ep["has_photo"].sum())}
    pd.Series(photos).to_csv(out / "table_photos.csv", header=["count"])
    summary["photos"] = photos

    # ------------------------------------------- meal responses by group
    log("meal responses by group")
    resp_tables = []
    curve_summ = {}
    for d in io.DEVICES:
        ok = meals.eligible(ep, d)
        sub = ep[ok]
        per = sub.groupby("pid").agg(median_iauc=(f"{d}_iauc", "median"),
                                     median_peak_rise=(f"{d}_peak_rise", "median"),
                                     median_time_to_peak=(f"{d}_time_to_peak", "median"),
                                     n_meals=(f"{d}_iauc", "size")).join(people["group"])
        t = stats.compare_groups(per, ["median_iauc", "median_peak_rise", "median_time_to_peak", "n_meals"])
        t.insert(0, "carb_bin", "all")
        t.insert(0, "device", d)
        resp_tables.append(t)
        glyc[d] = glyc[d].join(per[["median_iauc", "median_peak_rise", "median_time_to_peak"]])
        for lo, hi in CARB_BINS:
            label = f"{lo}-{hi}"
            m = ok & (ep["carbs"] >= lo) & (ep["carbs"] < hi)
            sb = ep[m]
            per_b = sb.groupby("pid").agg(median_iauc=(f"{d}_iauc", "median"),
                                          median_peak_rise=(f"{d}_peak_rise", "median"),
                                          carbs_median=("carbs", "median")).join(people["group"])
            tb = stats.compare_groups(per_b, ["median_iauc", "median_peak_rise", "carbs_median"])
            tb.insert(0, "carb_bin", label)
            tb.insert(0, "device", d)
            resp_tables.append(tb)
            # average curves: participant mean curve first, then mean across participants
            cv = pd.DataFrame(curves[d][m.to_numpy()], index=sb.index)
            cv["pid"] = sb["pid"].to_numpy()
            pc = cv.groupby("pid").mean()
            pc_n = cv.groupby("pid").size()
            for g in io.GROUPS:
                pids = [pid for pid in pc.index if people.loc[pid, "group"] == g]
                if len(pids) < 3:
                    continue
                arr = pc.loc[pids].to_numpy(float)
                mean = np.nanmean(arr, 0)
                n_eff = np.sum(np.isfinite(arr), 0)
                se = np.nanstd(arr, 0, ddof=1) / np.sqrt(n_eff)
                tcrit = sps.t.ppf(0.975, np.maximum(n_eff - 1, 1))
                if d == PRIMARY:
                    curve_summ[(label, g)] = (mean, mean - tcrit * se, mean + tcrit * se, len(pids),
                                              int(pc_n.loc[pids].sum()))
    resp = pd.concat(resp_tables, ignore_index=True)
    fmt_table(resp, out / "table_meal_response_by_group.csv")
    plots.group_curves(curve_summ, meals.CURVE_GRID, [f"{lo}-{hi}" for lo, hi in CARB_BINS],
                       out / "figures" / "fig2_postprandial_curves_by_group.png", DEVICE_LABEL[PRIMARY])

    # ------------------------------------------------------- mixed model
    log("mixed models")
    base = pd.concat([ep[["pid", "group", "time", "meal_type"]], F], axis=1)
    mm_out = {}
    specs = {"primary_dexcom_sqrt": (PRIMARY, "", "sqrt"),
             "sensitivity_dexcom_log1p": (PRIMARY, "", "log1p"),
             "sensitivity_dexcom_sqrt_plus_baseline": (PRIMARY, " + pre_glucose", "sqrt"),
             "replication_libre_sqrt": ("libre", "", "sqrt")}
    coef_all, slopes_all, pred_all = [], [], []
    for name, (d, extra, tr) in specs.items():
        dat = base[meals.eligible(ep, d)].copy()
        dat["iauc"] = ep.loc[dat.index, f"{d}_iauc"]
        if d != PRIMARY:  # baseline from the same device as the outcome
            dat["pre_glucose"] = ep.loc[dat.index, f"{d}_baseline"]
        fit = models.fit_mixed(dat, "iauc", extra=extra, transform=tr)
        ct = models.coef_table(fit)
        ct.insert(0, "model", name)
        coef_all.append(ct.reset_index().rename(columns={"index": "term"}))
        slopes = models.group_carb_slopes(fit)
        slopes.insert(0, "model", name)
        slopes_all.append(slopes)
        pr = models.predicted_iauc(fit)
        pr.insert(0, "model", name)
        pred_all.append(pr)
        mfit = fit["model"]
        mm_out[name] = {
            "n_meals": fit["n_meals"], "n_participants": fit["n_participants"], "transform": tr,
            "random_carb_slope": fit["random_slope"], "lr_stat_slope": float(fit["lr_stat"]),
            "lr_p_slope": float(fit["lr_p"]), "converged": fit["converged"], "optimizer": fit["optimizer"],
            "random_effects_cov": mfit.cov_re.round(5).to_dict(), "residual_var": float(mfit.scale),
            "resid_skew": float(sps.skew(mfit.resid)), "fit_warnings": fit["warnings"],
            "interaction_terms": ct.loc[[t for t in ct.index if t.startswith("carbs10:")],
                                        ["estimate", "ci_low", "ci_high", "p"]].to_dict("index"),
        }
        if name == "primary_dexcom_sqrt":
            raw = models.fit_mixed(dat, "iauc", extra=extra, transform="none")
            mm_out[name]["resid_skew_untransformed"] = float(sps.skew(raw["model"].resid))
            plots.coef_forest(models.coef_table(fit), out / "figures" / "fig3_mixed_model_coefficients.png",
                              f"Linear mixed model of sqrt(iAUC), {DEVICE_LABEL[PRIMARY]}\n"
                              f"{fit['n_meals']} meals, {fit['n_participants']} participants; "
                              f"random intercept{' + carb slope' if fit['random_slope'] else ''}",
                              xlabel="Coefficient on sqrt(iAUC [mg/dL*h]), 95% CI")
    pd.concat(coef_all).to_csv(out / "table_mixed_model_coefficients.csv", index=False, float_format="%.4g")
    pd.concat(slopes_all).to_csv(out / "table_mixed_model_carb_slopes.csv", index=False, float_format="%.4g")
    pd.concat(pred_all).to_csv(out / "table_mixed_model_predicted_iauc.csv", index=False, float_format="%.4g")
    summary["mixed_model"] = mm_out

    # ------------------------------------------- prediction / ablation
    log("prediction with modality ablation")
    ok = meals.eligible(ep, PRIMARY)
    D = base[ok].reset_index(drop=True)
    Y = {t: ep.loc[ok, f"{PRIMARY}_{t}"].to_numpy(float) for t in TARGETS}
    groups = D["pid"].to_numpy()
    train_mask = predict.temporal_split(D)
    summary["prediction_n"] = {"meals": int(len(D)), "participants": int(D["pid"].nunique()),
                               "temporal_train": int(train_mask.sum()), "temporal_test": int((~train_mask).sum())}
    rows = []
    steps = list(features.ABLATION_STEPS)
    for tname in TARGETS:
        y = Y[tname]
        hist = predict.personal_history(y, groups, train_mask)
        for model in ["ridge", "gbm"]:
            for setting in ["LOSO (new person)", "temporal (same person, later meals)"]:
                these = steps + ([("+ personal history", None)] if setting.startswith("temporal") else [])
                these = these + [("secondary: + post-meal activity", "post")]
                for step, grp in these:
                    if grp is None:
                        X = D[features.columns_for(steps[-1][1])].assign(personal_history=hist)
                    elif grp == "post":
                        X = D[features.columns_for(steps[-1][1] + ["activity_post"])]
                    else:
                        X = D[features.columns_for(grp)]
                    if setting.startswith("LOSO"):
                        yhat = predict.run_loso(X, y, groups, model)
                        m = predict.metrics(y, yhat, groups)
                        # same predictions scored only on the temporal test meals, so the
                        # two settings can be compared on identical meals
                        te = ~train_mask
                        m2 = predict.metrics(y[te], yhat[te], groups[te])
                        rows.append({"target": tname, "setting": "LOSO scored on temporal test meals",
                                     "model": model, "step": step, **m2})
                    else:
                        yhat = predict.run_temporal(X, y, groups, train_mask, model)
                        te = ~train_mask
                        m = predict.metrics(y[te], yhat[te], groups[te])
                    rows.append({"target": tname, "setting": setting, "model": model, "step": step, **m})
                    log(f"  {tname} {model} {setting} {step}: r={m['pooled_r']:.3f} MAE={m['pooled_mae']:.1f}")
    abl = pd.DataFrame(rows)
    abl.to_csv(out / "table_prediction_ablation.csv", index=False, float_format="%.4g")
    main_abl = abl[~abl["step"].str.startswith("secondary") & ~abl["setting"].str.contains("scored on")]
    plots.ablation(main_abl.replace({"target": {"iauc": "2-h iAUC", "peak_rise": "Peak rise"}}),
                   out / "figures" / "fig4_modality_ablation_r.png")
    plots.ablation(main_abl.replace({"target": {"iauc": "2-h iAUC", "peak_rise": "Peak rise"}}),
                   out / "figures" / "fig5_modality_ablation_per_participant_r.png", metric="pp_r_median",
                   ylabel="Median within-participant r")

    # ------------------------------------------- group classification
    log("participant-level group classification")
    wear = pd.DataFrame({pid: classify.wearable_features(p.ts) for pid, p in parts.items()}).T
    P = wear.join(glyc[PRIMARY][classify.CGM_FEATURES]).join(people["group"])
    y = P["group"].to_numpy()
    clf_rows, cms = [], {}
    for name, cols in {"wearable_only": classify.WEARABLE_FEATURES, "cgm_derived": classify.CGM_FEATURES,
                       "wearable_plus_cgm": classify.WEARABLE_FEATURES + classify.CGM_FEATURES}.items():
        r = classify.evaluate(P[cols], y, n_perm=args.n_perm)
        cms[name] = pd.DataFrame(r.pop("confusion"), index=[f"true_{g}" for g in classify.LABELS],
                                 columns=[f"pred_{g}" for g in classify.LABELS])
        clf_rows.append({"feature_set": name, **r})
        log(f"  {name}: balanced acc={r['balanced_accuracy']:.3f} p={r['perm_p']:.3f}")
    clf = pd.DataFrame(clf_rows)
    clf.to_csv(out / "table_group_classification.csv", index=False, float_format="%.4g")
    pd.concat(cms, names=["feature_set", "true"]).to_csv(out / "table_group_classification_confusion.csv")
    summary["classification"] = clf.to_dict("records")
    summary["wearable_by_group"] = stats.compare_groups(P, classify.WEARABLE_FEATURES).set_index("metric")[
        ["healthy_median", "prediabetes_median", "t2d_median", "kw_p", "eta2_H"]].to_dict("index")
    stats.compare_groups(P, classify.WEARABLE_FEATURES).to_csv(out / "table_wearable_by_group.csv", index=False,
                                                                float_format="%.4g")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("done")


if __name__ == "__main__":
    main()
