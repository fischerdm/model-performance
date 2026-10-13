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
    average_precision,
    calibration_bins,
    gini_index,
    isotonic_recalibration,
    lorenz_curve,
    pr_points,
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
MODEL_COLORS = {"GLM": "#2a78d6", "GBM": "#eb6834", "Blend": "#1baf7a", "Null": MUTED}
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


def plot_roc(y, preds: dict, ax=None, operating_points: dict | None = None, colors: dict | None = None):
    """ROC curves for several models, with AUC in the legend.

    ``operating_points`` maps a label to ``(model name, threshold)`` and marks
    where that threshold sits on the model's curve. The random-guessing
    diagonal is drawn unless the Null model (which traces it) is plotted.
    """
    ax = _ax(ax)
    colors = {**MODEL_COLORS, **(colors or {})}
    if "Null" not in preds:
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="Random guessing (AUC 0.5)")
    for name, p in preds.items():
        roc = roc_points(y, p)
        auc = np.trapezoid(roc["tpr"], roc["fpr"])
        ax.plot(roc["fpr"], roc["tpr"], color=colors.get(name), label=f"{name} (AUC {auc:.3f})")
    for label, (name, t) in (operating_points or {}).items():
        m = threshold_metrics(y, preds[name], t)
        ax.plot(m["false positive rate"], m["recall (TPR)"], "o", markersize=8,
                color=colors.get(name), markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, (m["false positive rate"], m["recall (TPR)"]), xytext=(8, -4),
                    textcoords="offset points", va="top", fontsize=9, color=INK_SECONDARY)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    return ax


def plot_pr(y, preds: dict, ax=None, iso_f1=(0.1, 0.2, 0.3), colors: dict | None = None,
            labels: dict | None = None, baseline: bool = True, ylim: float = 1.0):
    """Precision-recall curves with average precision (AP) in the legend.

    The horizontal line is the positive rate: the precision of random
    flagging (and the AP of the Null model). Thin gray curves mark points of
    equal F1 score.
    """
    ax = _ax(ax)
    colors = {**MODEL_COLORS, **(colors or {})}
    labels = labels or {}
    y = np.asarray(y)
    r = np.linspace(0.001, 1, 400)
    for f in iso_f1:
        prec = f * r / (2 * r - f)
        keep = (prec > 0) & (prec <= ylim)
        ax.plot(r[keep], prec[keep], color=GRID, linewidth=1, zorder=0)
        ax.annotate(f"F1 = {f:g}", (1, f / (2 - f)), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=8, color=MUTED)
    if baseline:
        rate = y.mean()
        ax.axhline(rate, color=MUTED, linewidth=1, label=f"Null / random (AP {rate:.3f})")
    for name, p in preds.items():
        pr = pr_points(y, p)
        ap = average_precision(y, p)
        ax.plot(np.r_[0, pr["recall"]], np.r_[pr["precision"].iloc[0], pr["precision"]],
                drawstyle="steps-pre", color=colors.get(name), label=f"{labels.get(name, name)} (AP {ap:.3f})")
    ax.set_xlabel("Recall (share of claims flagged)")
    ax.set_ylabel("Precision (share of flagged with a claim)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ylim)
    ax.legend(loc="upper right")
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


# --- Ranking and segmentation ----------------------------------------------


def plot_segments(table, models, ax=None, ratio: bool = False, xlabel: str = "Decile",
                  ylabel: str | None = None, title: str | None = None, colors: dict | None = None,
                  legend_loc: str = "upper left"):
    """Lift or double lift chart from a ``segment_table``.

    By default shows the actual average per bin (with its 95% interval) next to
    each model's average prediction. With ``ratio=True`` it shows actual ÷
    predicted per model instead: 1 means the model's average is right in that
    bin (an insurer would call it a loss ratio of 100%).
    """
    ax = _ax(ax)
    colors = {**MODEL_COLORS, **(colors or {})}
    x = table.index.to_numpy()
    marker = dict(marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    if ratio:
        ax.axhline(1, color=MUTED, linewidth=1)
        offsets = np.linspace(-0.12, 0.12, len(models)) if len(models) > 1 else [0.0]
        for name, dx in zip(models, offsets):
            r = table["actual"] / table[name]
            ax.errorbar(x + dx, r, yerr=[r - table["lower"] / table[name], table["upper"] / table[name] - r],
                        color=colors.get(name), linewidth=1.5, elinewidth=1, capsize=0, label=f"Actual ÷ {name}",
                        **marker)
        ax.set_ylabel(ylabel or "Actual ÷ predicted")
        # Intervals span most of the height, so the legend goes above the plot.
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models))
    else:
        ax.errorbar(x, table["actual"], yerr=[table["actual"] - table["lower"], table["upper"] - table["actual"]],
                    color=INK_SECONDARY, linestyle="none", elinewidth=1, capsize=0, label="Actual (95% interval)",
                    **marker)
        for name in models:
            ax.plot(x, table[name], color=colors.get(name), label=f"{name} predicted", **marker)
        ax.set_ylabel(ylabel or "Average per policy-year (EUR)")
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.05 * table["upper"].max()))
        ax.legend(loc=legend_loc)
    ax.set_xticks(x)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return ax


def plot_lorenz(y, preds: dict, weight, ax=None, oracle: bool = True, colors: dict | None = None):
    """Lorenz curves (share of claim cost against share of exposure, lowest predictions first).

    The diagonal is a model that doesn't separate risks (Null); the dashed
    'oracle' curve sorts by the actual outcome, the best any model could do.
    Gini indices are in the legend.
    """
    ax = _ax(ax)
    colors = {**MODEL_COLORS, **(colors or {})}
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="Null / no separation (Gini 0)")
    if oracle:
        lc = lorenz_curve(y, y, weight)
        ax.plot(lc["exposure_share"], lc["loss_share"], color=INK_SECONDARY, linewidth=1.5, linestyle="--",
                label=f"Oracle: sorted by actual (Gini {gini_index(y, y, weight):.3f})")
    for name, p in preds.items():
        lc = lorenz_curve(y, p, weight)
        ax.plot(lc["exposure_share"], lc["loss_share"], color=colors.get(name),
                label=f"{name} (Gini {gini_index(y, p, weight):.3f})")
    ax.set_xlabel("Share of exposure (lowest predicted risk first)")
    ax.set_ylabel("Share of claim cost")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    return ax


# --- Model comparison --------------------------------------------------------


def plot_paired_bootstrap(boot, a: str, b: str, axes, metric: str = "AUC", level: float = 0.95):
    """Paired bootstrap of two models: one resample per point, and the distribution of the difference.

    Left: each resample's metric for model ``b`` against model ``a``; points
    above the diagonal are resamples where ``a`` wins. Right: histogram of
    ``a − b`` with its percentile interval shaded and 0 (no difference) marked.
    """
    left, right = axes
    lo, hi = np.quantile(np.r_[boot[a], boot[b]], [0, 1])
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    left.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    left.scatter(boot[b], boot[a], s=8, color=INK_SECONDARY, alpha=0.35, linewidths=0)
    left.annotate(f"{a} better", (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=9, color=MUTED)
    left.annotate(f"{b} better", (0.96, 0.04), xycoords="axes fraction", ha="right", fontsize=9, color=MUTED)
    left.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal", xlabel=f"{metric} of {b}", ylabel=f"{metric} of {a}")
    left.set_title("Each point: one resample")

    diff = boot[a] - boot[b]
    tail = (1 - level) / 2
    q_lo, q_hi = np.quantile(diff, [tail, 1 - tail])
    right.axvspan(q_lo, q_hi, color=GRID, alpha=0.6, linewidth=0, label=f"{level:.0%} interval")
    counts, *_ = right.hist(diff, bins=40, histtype="step", linewidth=2, color=INK_SECONDARY)
    right.set_ylim(0, 1.2 * counts.max())  # headroom for the labels
    right.axvline(0, color=INK, linewidth=1)
    right.annotate("no difference", (0, 1), xycoords=("data", "axes fraction"), xytext=(4, -4),
                   textcoords="offset points", va="top", fontsize=9, color=INK_SECONDARY)
    right.set(xlabel=f"{metric} of {a} − {metric} of {b}", ylabel="Resamples")
    right.set_title("The paired difference")
    right.legend(loc="upper right")
    return axes


def plot_intervals(table, ax=None, xlabel: str = "Difference", title: str | None = None):
    """Estimates with confidence intervals, one row per comparison (a 'forest plot').

    ``table`` has one row per comparison (labelled by its index) and the
    columns ``difference``, ``lower`` and ``upper``. An interval that crosses
    the vertical line at 0 is compatible with no difference.
    """
    ax = _ax(ax)
    rows = np.arange(len(table))[::-1]  # first row on top
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.errorbar(table["difference"], rows,
                xerr=[table["difference"] - table["lower"], table["upper"] - table["difference"]],
                color=INK_SECONDARY, linestyle="none", elinewidth=2, capsize=0, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_yticks(rows, table.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    return ax


# --- Drift -------------------------------------------------------------------

PSI_THRESHOLDS = {"moderate shift": 0.1, "major shift": 0.25}


def plot_psi(psi, ax=None, title: str | None = None, xmax: float | None = None):
    """PSI per feature as horizontal bars, largest on top, with the rule-of-thumb thresholds."""
    ax = _ax(ax)
    psi = psi.sort_values()
    rows = np.arange(len(psi))
    ax.barh(rows, psi, height=0.6, color=INK_SECONDARY)
    ax.set_yticks(rows, psi.index)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    xmax = xmax or max(0.3, 1.1 * psi.max())
    for label, x in PSI_THRESHOLDS.items():
        ax.axvline(x, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(label, (x, 0.02), xycoords=("data", "axes fraction"), xytext=(3, 0),
                    textcoords="offset points", rotation=90, va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("PSI")
    if title:
        ax.set_title(title)
    return ax


def plot_psi_bins(table, ax=None, title: str | None = None, xlabel: str = "Bin"):
    """Share of reference and current data per bin, from a ``psi_table``."""
    ax = _ax(ax)
    x = np.arange(len(table))
    ax.bar(x - 0.2, table["reference"], width=0.4, color=MUTED, label="Reference")
    ax.bar(x + 0.2, table["current"], width=0.4, color=METRIC_COLORS[1], label="Current")
    ax.set_xticks(x, table.index, rotation=45, ha="right")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of policies")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    return ax


def plot_over_time(table, ax=None, ylabel: str = "", hlines: dict | None = None,
                   title: str | None = None, ylim=None):
    """One line per column of ``table`` (index: month), labelled at its last point."""
    ax = _ax(ax)
    for label, y in (hlines or {}).items():
        ax.axhline(y, color=MUTED, linewidth=1, linestyle="--")
        ax.annotate(label, (0, y), xycoords=("axes fraction", "data"), xytext=(3, 3),
                    textcoords="offset points", fontsize=8, color=MUTED)
    x = table.index.to_numpy()
    for (name, values), color in zip(table.items(), METRIC_COLORS):
        ax.plot(x, values, color=color, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1, label=name)
    ax.set_xticks(x)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title)
    return ax


def plot_estimated_vs_realized(table, metric: str, ax=None, reference: float | None = None,
                               title: str | None = None, color: str = MODEL_COLORS["GBM"]):
    """Monitoring chart: a metric estimated without labels against its realized value per month.

    ``table`` (index: month) has the columns ``estimated``, ``realized``,
    ``lower`` and ``upper`` (the realized value's 95% interval).
    """
    ax = _ax(ax)
    x = table.index.to_numpy()
    if reference is not None:
        ax.axhline(reference, color=MUTED, linewidth=1, linestyle="--", label="Reference")
    ax.errorbar(x, table["realized"], yerr=[table["realized"] - table["lower"], table["upper"] - table["realized"]],
                color=INK_SECONDARY, linestyle="none", elinewidth=1, capsize=0, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label="Realized (95% interval)")
    ax.plot(x, table["estimated"], color=color, label="Estimated (CBPE)")
    ax.set_xticks(x[::2])
    ax.set_xlabel("Month")
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    return ax
