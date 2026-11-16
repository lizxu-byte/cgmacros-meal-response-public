"""Linear mixed model of meal iAUC.

Outcome scale.  iAUC (mg/dL x h) is right-skewed and ~9 % of eligible meals
have iAUC < 1 (no rise above baseline).  On the real data the residual
skewness of the model was 0.89 untransformed, -0.88 for log(iAUC + 1) (the
near-zero responses form a long left tail) and -0.09 for sqrt(iAUC), so the
primary outcome is ``sqrt(iAUC)`` and log(iAUC + 1) is a sensitivity
analysis (``transform="log1p"``).  run_all recomputes and reports these
skewness values.  Because sqrt-scale coefficients are hard to read, predicted
iAUC by group at fixed carbohydrate loads (other covariates at their means,
random effects at zero, back-transformed by squaring) is also reported.

Fixed effects (all continuous predictors grand-mean centred):
  carbs (per 10 g) x group, protein (per 10 g), fat (per 10 g),
  fibre (per 5 g), pre-meal HR (per 10 bpm), pre-meal activity energy
  (kcal/min), time of day (sin, cos), log minutes since previous meal.
Group reference: healthy.
Random effects per participant: intercept and carbohydrate slope (kept if the
model converges and a likelihood-ratio test against intercept-only favours it;
the LRT p-value uses the 50:50 mixture of chi2(1) and chi2(2) because the
variance parameter is on the boundary under H0).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

SCALES = {"carbs": 10.0, "protein": 10.0, "fat": 10.0, "fiber": 5.0, "hr_pre": 10.0}
COVARIATES = ["carbs10", "protein10", "fat10", "fiber5", "hr_pre10", "activity_kcal_pre",
              "hour_sin", "hour_cos", "log_min_since_prev"]


TRANSFORMS = {
    "sqrt": (lambda x: np.sqrt(np.clip(x, 0, None)), lambda z: np.clip(z, 0, None) ** 2),
    "log1p": (lambda x: np.log1p(np.clip(x, 0, None)), np.expm1),
    "none": (lambda x: x, lambda z: z),
}


def prepare(df: pd.DataFrame, outcome: str, transform: str = "sqrt") -> pd.DataFrame:
    d = df.copy()
    d["y"] = TRANSFORMS[transform][0](d[outcome].astype(float))
    d["carbs10"] = d["carbs"] / 10
    d["protein10"] = d["protein"] / 10
    d["fat10"] = d["fat"] / 10
    d["fiber5"] = d["fiber"] / 5
    d["hr_pre10"] = d["hr_pre"] / 10
    cols = ["y", "pid", "group", "pre_glucose"] + COVARIATES
    d = d[cols].dropna()
    for c in COVARIATES + ["pre_glucose"]:
        d[c] = d[c] - d[c].mean()
    d["group"] = pd.Categorical(d["group"], categories=["healthy", "prediabetes", "t2d"])
    return d


def _formula(extra: str = "") -> str:
    rhs = "carbs10 * C(group, Treatment('healthy')) + " + " + ".join(COVARIATES[1:])
    return f"y ~ {rhs}{extra}"


OPTIMIZERS = ("lbfgs", "bfgs", "powell", "nm")


def _fit(formula, d, re_formula, reml):
    """Fit with several optimizers; keep the converged fit with the highest finite
    log-likelihood.  L-BFGS alone sometimes stops at a degenerate point (infinite
    log-likelihood with zero random-effect variance, or a singular Hessian)."""
    model = smf.mixedlm(formula, d, groups=d["pid"], re_formula=re_formula)
    best, best_name = None, None
    for name in OPTIMIZERS:
        try:
            m = model.fit(reml=reml, method=name)
        except (np.linalg.LinAlgError, ValueError):
            continue
        if not (m.converged and np.isfinite(m.llf)):
            continue
        if best is None or m.llf > best.llf + 1e-6:
            best, best_name = m, name
    if best is None:
        raise RuntimeError("no optimizer produced a converged fit with finite log-likelihood")
    return best, best_name


def fit_mixed(df: pd.DataFrame, outcome: str = "iauc", extra: str = "", transform: str = "sqrt") -> dict:
    """Fit intercept-only and intercept+carb-slope models; return the chosen fit."""
    d = prepare(df, outcome, transform)
    f = _formula(extra)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m0, _ = _fit(f, d, None, reml=False)
        m1, _ = _fit(f, d, "~carbs10", reml=False)
        lr = 2 * (m1.llf - m0.llf)
        p_lr = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2) if lr > 0 else 1.0
        use_slope = bool(m1.converged and p_lr < 0.05)
        final, optimizer = _fit(f, d, "~carbs10" if use_slope else None, reml=True)
        msgs = sorted({str(x.message) for x in w})
    return dict(model=final, data=d, random_slope=use_slope, lr_stat=lr, lr_p=p_lr, transform=transform,
                n_meals=len(d), n_participants=int(d["pid"].nunique()), warnings=msgs,
                converged=bool(final.converged), optimizer=optimizer, outcome_mean=float(df[outcome].mean()),
                carbs_mean=float(df.loc[d.index, "carbs"].mean()))


def coef_table(fit) -> pd.DataFrame:
    m = fit["model"] if isinstance(fit, dict) else fit
    ci = m.conf_int()
    t = pd.DataFrame({"estimate": m.fe_params, "se": m.bse_fe,
                      "ci_low": ci.loc[m.fe_params.index, 0], "ci_high": ci.loc[m.fe_params.index, 1],
                      "p": m.pvalues.loc[m.fe_params.index]})
    t.index = [clean_term(i) for i in t.index]
    return t


def clean_term(name: str) -> str:
    return (name.replace("C(group, Treatment('healthy'))[T.prediabetes]", "group[prediabetes]")
                .replace("C(group, Treatment('healthy'))[T.t2d]", "group[t2d]"))


def group_carb_slopes(fit) -> pd.DataFrame:
    """Carbohydrate slope (per 10 g, log scale) within each group, with 95 % CI."""
    m = fit["model"] if isinstance(fit, dict) else fit
    names = list(m.fe_params.index)
    cov = m.cov_params().loc[names, names].to_numpy()
    beta = m.fe_params.to_numpy()
    rows = []
    for g in ["healthy", "prediabetes", "t2d"]:
        L = np.zeros(len(names))
        L[names.index("carbs10")] = 1
        if g != "healthy":
            L[names.index(f"carbs10:C(group, Treatment('healthy'))[T.{g}]")] = 1
        est = L @ beta
        se = float(np.sqrt(L @ cov @ L))
        rows.append({"group": g, "slope_per10g": est, "se": se,
                     "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
                     "p": float(2 * stats.norm.sf(abs(est / se)))})
    return pd.DataFrame(rows)


def predicted_iauc(fit, carbs_grams=(30, 60, 90)) -> pd.DataFrame:
    """Back-transformed predicted iAUC by group and carbohydrate load.

    Other covariates at their sample means (0 after centring), random effects
    at zero; 95 % CI from the fixed-effect covariance on the model scale,
    back-transformed (so these describe a typical participant, not the mean
    over participants).
    """
    m = fit["model"]
    inv = TRANSFORMS[fit["transform"]][1]
    names = list(m.fe_params.index)
    cov = m.cov_params().loc[names, names].to_numpy()
    beta = m.fe_params.to_numpy()
    rows = []
    for g in ["healthy", "prediabetes", "t2d"]:
        for c in carbs_grams:
            x = np.zeros(len(names))
            x[names.index("Intercept")] = 1
            cc = (c - fit["carbs_mean"]) / 10
            x[names.index("carbs10")] = cc
            if g != "healthy":
                x[names.index(f"C(group, Treatment('healthy'))[T.{g}]")] = 1
                x[names.index(f"carbs10:C(group, Treatment('healthy'))[T.{g}]")] = cc
            est = x @ beta
            se = float(np.sqrt(x @ cov @ x))
            rows.append({"group": g, "carbs_g": c, "pred_iauc": float(inv(est)),
                         "ci_low": float(inv(est - 1.96 * se)), "ci_high": float(inv(est + 1.96 * se))})
    return pd.DataFrame(rows)
