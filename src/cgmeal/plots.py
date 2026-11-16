"""Figures (matplotlib, static PNG). Only aggregated data are drawn."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GROUP_COLORS = {"healthy": "#2a78d6", "prediabetes": "#eb6834", "t2d": "#1baf7a"}
GROUP_LABELS = {"healthy": "Healthy", "prediabetes": "Prediabetes", "t2d": "Type 2 diabetes"}
MODEL_COLORS = {"ridge": "#2a78d6", "gbm": "#eb6834"}
MODEL_LABELS = {"ridge": "Ridge regression", "gbm": "Gradient boosting"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bland_altman(mean_vals, diff_vals, summary: dict, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    hb = ax.hexbin(mean_vals, diff_vals, gridsize=60, mincnt=5, cmap="Blues", bins="log", linewidths=0)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("paired readings per cell (log, cells < 5 hidden)", fontsize=8, color=INK2)
    for y, lab, ls in [(summary["bias_mgdl"], "bias", "-"), (summary["loa_low"], "-1.96 SD", "--"),
                       (summary["loa_high"], "+1.96 SD", "--")]:
        ax.axhline(y, color=INK, lw=1.2, ls=ls)
        ax.annotate(f"{lab} {y:.1f}", xy=(0.01, y), xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", va="bottom", ha="left", fontsize=8, color=INK,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.axhline(0, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("Mean of Libre Pro and Dexcom G6 (mg/dL)", color=INK)
    ax.set_ylabel("Libre Pro minus Dexcom G6 (mg/dL)", color=INK)
    ax.set_title(f"CGM agreement: {summary['n_pairs']:,} paired readings, {summary['n_participants']} participants",
                 fontsize=10, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def group_curves(curve_summary: dict, grid, bins, path, device_label: str):
    """curve_summary[(bin_label, group)] = (mean, ci_low, ci_high, n_participants, n_meals)."""
    fig, axes = plt.subplots(1, len(bins), figsize=(4.2 * len(bins), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bins):
        for g in GROUP_COLORS:
            key = (b, g)
            if key not in curve_summary:
                continue
            m, lo, hi, n_p, n_m = curve_summary[key]
            ax.fill_between(grid, lo, hi, color=GROUP_COLORS[g], alpha=0.15, lw=0)
            ax.plot(grid, m, color=GROUP_COLORS[g], lw=2, label=f"{GROUP_LABELS[g]} (n={n_p}, {n_m} meals)")
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"Carbohydrate {b} g", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Minutes from meal start", color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)
        _style(ax)
    axes[0].set_ylabel(f"Glucose change from baseline (mg/dL)\n{device_label}", color=INK)
    fig.suptitle("Mean postprandial curves by health state (participant means; band = 95% CI across participants)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def coef_forest(table: pd.DataFrame, path, title: str, xlabel: str = "Coefficient, 95% CI"):
    t = table.drop(index="Intercept", errors="ignore").iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(t) + 1.2))
    y = np.arange(len(t))
    ax.errorbar(t["estimate"], y, xerr=[t["estimate"] - t["ci_low"], t["ci_high"] - t["estimate"]],
                fmt="o", color="#2a78d6", ecolor="#2a78d6", ms=5, lw=1.5, capsize=0)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(t.index, fontsize=8, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ablation(df: pd.DataFrame, path, metric: str = "pooled_r", ylabel: str = "Pearson r (pooled test meals)"):
    """df rows: setting, target, model, step, metric columns."""
    targets = list(df["target"].unique())
    settings = list(df["setting"].unique())
    fig, axes = plt.subplots(len(targets), len(settings), figsize=(5.6 * len(settings), 3.4 * len(targets)),
                             sharey=True, squeeze=False)
    for i, tg in enumerate(targets):
        for j, st in enumerate(settings):
            ax = axes[i, j]
            sub = df[(df["target"] == tg) & (df["setting"] == st)]
            steps = list(dict.fromkeys(sub["step"]))
            x = np.arange(len(steps))
            w = 0.38
            for k, mdl in enumerate(["ridge", "gbm"]):
                s = sub[sub["model"] == mdl].set_index("step").reindex(steps)
                ax.bar(x + (k - 0.5) * w, s[metric], width=w - 0.03, color=MODEL_COLORS[mdl],
                       label=MODEL_LABELS[mdl])
            ax.set_xticks(x)
            ax.set_xticklabels(steps, rotation=30, ha="right", fontsize=8)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_title(f"{tg}  |  {st}", fontsize=10, color=INK, loc="left")
            _style(ax)
            if j == 0:
                ax.set_ylabel(ylabel, color=INK)
    axes[0, -1].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
