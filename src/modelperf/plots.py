"""Consistent matplotlib charts for the documentation.

Colors follow a validated, colorblind-safe palette. Each model keeps its color
on every page (GLM blue, GBM orange, Null gray reference), so readers can
recognise it without re-reading legends.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .metrics import (
    calibration_bins,
    isotonic_recalibration,
    reliability_band,
    roc_points,
    threshold_metrics,
)

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed entity colors
MODEL_COLORS = {"GLM": "#2a78d6", "GBM": "#eb6834", "Null": MUTED}
CLASS_COLORS = {0: "#1baf7a", 1: "#4a3aa7"}  # outcome 0 / 1
METRIC_COLORS = ["#1baf7a", "#4a3aa7", "#e87ba4"]  # always direct-labelled

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def use_style() -> None:
    """Apply the project's chart style to all subsequent matplotlib figures."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def _ax(ax):
    return ax if ax is not None else plt.subplots(figsize=(5.5, 4))[1]


# --- Binary outcomes -------------------------------------------------------


def plot_confusion_matrix(counts: dict, ax=None, title: str | None = None):
    """2×2 confusion matrix: counts and share of all cases per cell."""
    ax = _ax(ax)
    cells = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    names = np.array([["TN", "FP"], ["FN", "TP"]])
    total = cells.sum()
    ax.imshow(np.log1p(cells), cmap=SEQUENTIAL, vmin=0, vmax=np.log1p(total))
    for i in range(2):
        for j in range(2):
            dark_fill = np.log1p(cells[i, j]) > 0.55 * np.log1p(total)
            ax.text(
                j, i, f"{names[i, j]}\n{cells[i, j]:,}\n{cells[i, j] / total:.1%}",
                ha="center", va="center", fontsize=10,
                color="white" if dark_fill else INK,
            )
    ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title)
    return ax


def plot_score_distributions(y, p, ax=None, threshold: float | None = None,
                             labels=("No claim", "Claim"), bins: int = 60):
    """Distribution of predicted probabilities for each actual class.

    Densities are shown (each class integrates to 1), so the rare class stays
    visible next to the common one.
    """
    ax = _ax(ax)
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, np.quantile(p, 0.999), bins + 1)
    for cls, label in zip((0, 1), labels):
        ax.hist(p[y == cls], bins=edges, density=True, histtype="step", linewidth=2,
                color=CLASS_COLORS[cls], label=f"{label} (n = {np.sum(y == cls):,})")
    if threshold is not None:
        ax.axvline(threshold, color=INK_SECONDARY, linewidth=1)
        ax.annotate(f"threshold {threshold:g}", (threshold, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Predicted probability of a claim")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    return ax


def plot_threshold_metrics(y, p, ax=None, metrics=("accuracy", "precision", "recall (TPR)"),
                           thresholds=None):
    """How threshold-based metrics change as the decision threshold moves."""
    ax = _ax(ax)
    if thresholds is None:
        thresholds = np.linspace(0.005, np.quantile(p, 0.995), 80)
    table = np.array([threshold_metrics(y, p, t)[list(metrics)].to_numpy() for t in thresholds])
    for k, (name, color) in enumerate(zip(metrics, METRIC_COLORS)):
        values = table[:, k]
        ax.plot(thresholds, values, color=color, label=name)
        last = np.flatnonzero(~np.isnan(values))[-1]
        ax.annotate(name, (thresholds[last], values[last]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(metrics))
    return ax


def plot_roc(y, preds: dict, ax=None, operating_points: dict | None = None):
    """ROC curves for several models, with AUC in the legend.

    ``operating_points`` maps a label to ``(model name, threshold)`` and marks
    where that threshold sits on the model's curve. The random-guessing
    diagonal is drawn unless the Null model (which traces it) is plotted.
    """
    ax = _ax(ax)
    if "Null" not in preds:
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="Random guessing (AUC 0.5)")
    for name, p in preds.items():
        roc = roc_points(y, p)
        auc = np.trapezoid(roc["tpr"], roc["fpr"])
        ax.plot(roc["fpr"], roc["tpr"], color=MODEL_COLORS.get(name), label=f"{name} (AUC {auc:.3f})")
    for label, (name, t) in (operating_points or {}).items():
        m = threshold_metrics(y, preds[name], t)
        ax.plot(m["false positive rate"], m["recall (TPR)"], "o", markersize=8,
                color=MODEL_COLORS.get(name), markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, (m["false positive rate"], m["recall (TPR)"]), xytext=(8, -4),
                    textcoords="offset points", va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax


def plot_reliability(y, preds: dict, ax=None, method: str = "corp", n_bins: int = 20,
                     strategy: str = "quantile", band: bool = False, max_p: float | None = None,
                     title: str | None = None, colors: dict | None = None):
    """Reliability diagram: observed rate against predicted probability.

    ``method="bins"`` plots one point per bin (with a 95% Wilson interval);
    ``method="corp"`` plots the isotonic (CORP) reliability curve, optionally
    with a 90% consistency band showing where a calibrated model's curve could
    fall by chance.
    """
    ax = _ax(ax)
    colors = {**MODEL_COLORS, **(colors or {})}
    y = np.asarray(y)
    if max_p is None:
        max_p = max(np.quantile(np.asarray(p), 0.995) for p in preds.values())
    ax.plot([0, max_p], [0, max_p], color=MUTED, linewidth=1)
    for name, p in preds.items():
        p = np.asarray(p)
        color = colors.get(name)
        if method == "bins":
            t = calibration_bins(y, p, n_bins=n_bins, strategy=strategy)
            ax.errorbar(t["mean_predicted"], t["observed_rate"],
                        yerr=[t["observed_rate"] - t["lower"], t["upper"] - t["observed_rate"]],
                        color=color, linewidth=1.5, elinewidth=1, capsize=0, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
        else:
            order = np.argsort(p)
            curve = isotonic_recalibration(y, p)[order]
            if band:
                grid = np.linspace(p.min(), min(p.max(), max_p), 200)
                b = reliability_band(p, grid)
                ax.fill_between(b["p"], b["lower"], b["upper"], color=color, alpha=0.12, linewidth=0,
                                label=f"{name}: 90% consistency band")
            ax.plot(p[order], curve, color=color, drawstyle="steps-post", label=name)
    ax.annotate("predictions too low", (0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=9, color=MUTED)
    ax.annotate("predictions too high", (0.97, 0.03), xycoords="axes fraction", ha="right",
                fontsize=9, color=MUTED)
    ax.set_xlim(0, max_p)
    ax.set_ylim(0, max_p)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    if title:
        ax.set_title(title)
    return ax


# --- Continuous outcomes ---------------------------------------------------


def plot_actual_vs_predicted(y, pred, ax=None, title: str | None = None, gridsize: int = 50):
    """Density of (predicted, actual) pairs with the line of perfect prediction."""
    ax = _ax(ax)
    y, pred = np.asarray(y), np.asarray(pred)
    lo, hi = np.quantile(np.r_[y, pred], [0.001, 0.999])
    ax.hexbin(pred, y, gridsize=gridsize, extent=(lo, hi, lo, hi), cmap=SEQUENTIAL,
              bins="log", mincnt=1, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, linewidth=1)  # perfect prediction
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if title:
        ax.set_title(title)
    return ax


def plot_residuals(y, pred, ax=None, title: str | None = None, gridsize: int = 50):
    """Residuals (actual − predicted) against the prediction."""
    ax = _ax(ax)
    y, pred = np.asarray(y), np.asarray(pred)
    resid = y - pred
    x_lo, x_hi = np.quantile(pred, [0.001, 0.999])
    r_max = np.quantile(np.abs(resid), 0.999)
    ax.hexbin(pred, resid, gridsize=gridsize, extent=(x_lo, x_hi, -r_max, r_max),
              cmap=SEQUENTIAL, bins="log", mincnt=1, linewidths=0)
    ax.axhline(0, color=INK_SECONDARY, linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual − predicted)")
    if title:
        ax.set_title(title)
    return ax
