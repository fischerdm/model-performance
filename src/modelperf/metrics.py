"""Performance metrics, implemented from their definitions.

scikit-learn provides most of these already. They are re-implemented here so
the documentation can show exactly what each metric computes; ``tests/``
checks every function against the scikit-learn reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm, rankdata
from sklearn.isotonic import IsotonicRegression

EPS = 1e-15

# --- Binary outcomes -------------------------------------------------------


def confusion_counts(y, p, threshold: float) -> dict[str, int]:
    """Counts of true/false positives/negatives when predicting 1 iff ``p >= threshold``."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(p) >= threshold
    return {
        "TP": int(np.sum(pred & y)),
        "FP": int(np.sum(pred & ~y)),
        "TN": int(np.sum(~pred & ~y)),
        "FN": int(np.sum(~pred & y)),
    }


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else np.nan


def threshold_metrics(y, p, threshold: float) -> pd.Series:
    """All confusion-matrix based metrics at one threshold."""
    c = confusion_counts(y, p, threshold)
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    n = tp + fp + tn + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return pd.Series(
        {
            "accuracy": (tp + tn) / n,
            "precision": precision,
            "recall (TPR)": recall,
            "specificity (TNR)": specificity,
            "false positive rate": 1 - specificity,
            "F1": _ratio(2 * precision * recall, precision + recall),
            "balanced accuracy": (recall + specificity) / 2,
            "predicted positive rate": (tp + fp) / n,
        },
        name=threshold,
    )


def roc_points(y, scores) -> pd.DataFrame:
    """ROC curve: one point per distinct score, from (0, 0) to (1, 1).

    Moving the threshold down past each distinct score turns all cases with
    that score into predicted positives; the curve records the resulting
    false positive rate and true positive rate.
    """
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    scores, y = scores[order], y[order]

    last_of_each_score = np.r_[np.flatnonzero(np.diff(scores)), y.size - 1]
    tps = np.cumsum(y)[last_of_each_score]
    fps = (last_of_each_score + 1) - tps
    return pd.DataFrame(
        {
            "threshold": np.r_[np.inf, scores[last_of_each_score]],
            "fpr": np.r_[0.0, fps / fps[-1]],
            "tpr": np.r_[0.0, tps / tps[-1]],
        }
    )


def auc_trapezoid(fpr, tpr) -> float:
    """Area under a curve given by its points (trapezoidal rule)."""
    return float(np.trapezoid(tpr, fpr))


def auc_rank(y, scores) -> float:
    """AUC as P(score of a random positive > score of a random negative).

    This is the Mann-Whitney U statistic scaled to [0, 1]; ties count 1/2.
    """
    y = np.asarray(y).astype(bool)
    ranks = rankdata(scores)  # average ranks for ties
    n_pos, n_neg = y.sum(), (~y).sum()
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def gini_from_auc(auc: float) -> float:
    """Gini coefficient (accuracy ratio) of a binary classifier: 2·AUC − 1."""
    return 2 * auc - 1


def pr_points(y, scores) -> pd.DataFrame:
    """Precision-recall curve: one point per distinct score, recall increasing.

    Uses the same threshold sweep as ``roc_points``; precision is
    TP / (TP + FP) among all cases scored at or above the threshold.
    """
    roc = roc_points(y, scores).iloc[1:]  # drop the 'flag nothing' point (precision undefined)
    y = np.asarray(y).astype(float)
    n_pos, n_neg = y.sum(), y.size - y.sum()
    tp, fp = roc["tpr"] * n_pos, roc["fpr"] * n_neg
    return pd.DataFrame(
        {"threshold": roc["threshold"], "recall": roc["tpr"], "precision": tp / (tp + fp)}
    ).reset_index(drop=True)


def average_precision(y, scores) -> float:
    """Average precision: precision at each threshold, weighted by the recall it adds.

    AP = Σ (R_k − R_{k−1}) · P_k. Unlike the trapezoidal area under the PR
    curve, it does not interpolate linearly between points, which would
    overstate precision.
    """
    pr = pr_points(y, scores)
    recall_gain = np.diff(np.r_[0.0, pr["recall"]])
    return float(np.sum(recall_gain * pr["precision"]))


def top_share_metrics(y, scores, share: float) -> pd.Series:
    """Precision, recall and lift when flagging the ``share`` of cases with the highest scores.

    Lift is precision divided by the overall positive rate: how many times
    more positives the flagged group contains than a random group of the
    same size.
    """
    y = np.asarray(y).astype(float)
    k = int(np.ceil(share * y.size))
    top = np.argsort(-np.asarray(scores, dtype=float), kind="mergesort")[:k]
    precision = y[top].mean()
    return pd.Series(
        {"precision": precision, "recall": y[top].sum() / y.sum(), "lift": precision / y.mean()},
        name=share,
    )


# --- Calibration and proper scoring rules ----------------------------------


def brier_score(y, p) -> float:
    """Mean squared difference between predicted probability and outcome."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    """Mean negative log-likelihood of the outcomes under the predicted probabilities."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


SCORES = {"Brier": brier_score, "log loss": log_loss}


def calibration_bins(y, p, n_bins: int = 10, strategy: str = "quantile", level: float = 0.95) -> pd.DataFrame:
    """Binned reliability table: mean prediction vs observed rate per bin.

    ``strategy`` is ``"quantile"`` (bins with equal counts) or ``"uniform"``
    (bins of equal width between the smallest and largest prediction). The
    observed rate gets a Wilson score interval.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    bin_id = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    table = (
        pd.DataFrame({"bin": bin_id, "p": p, "y": y})
        .groupby("bin")
        .agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
    )
    z = norm.ppf(0.5 + level / 2)
    n, r = table["n"], table["observed_rate"]
    centre = (r + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(r * (1 - r) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return table.assign(lower=centre - half, upper=centre + half).reset_index(drop=True)


def expected_calibration_error(y, p, n_bins: int = 10, strategy: str = "quantile") -> float:
    """ECE: share-weighted average gap between mean prediction and observed rate per bin."""
    t = calibration_bins(y, p, n_bins, strategy)
    return float(np.average(np.abs(t["mean_predicted"] - t["observed_rate"]), weights=t["n"]))


def calibration_intercept_slope(y, p) -> pd.Series:
    """Logistic recalibration: ``logit P(y=1) = a + b · logit(p)``.

    - Calibration intercept: ``a`` with ``b`` fixed at 1. Positive means the
      predictions are too low on average, negative too high.
    - Calibration slope: ``b`` from the two-parameter fit. Below 1 means the
      predictions are too extreme (spread too far from the mean), above 1 too
      timid.
    """
    y = np.asarray(y, dtype=float)
    z = logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))
    intercept = minimize_scalar(lambda a: log_loss(y, expit(a + z))).x
    _, b = minimize(lambda ab: log_loss(y, expit(ab[0] + ab[1] * z)), x0=[0.0, 1.0]).x
    return pd.Series({"calibration intercept": intercept, "calibration slope": b})


def isotonic_recalibration(y, p) -> np.ndarray:
    """Best monotone (non-decreasing) mapping of ``p`` to observed rates (PAV algorithm).

    This is the CORP reliability curve: evaluated at each prediction, it gives
    the observed rate among cases with similar predictions, with the bins
    chosen optimally by the data instead of by hand.
    """
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    return iso.fit(p, y).predict(p)


def murphy_decomposition(y, p, score: str = "Brier") -> pd.Series:
    """CORP decomposition: score = miscalibration (MCB) − discrimination (DSC) + uncertainty (UNC).

    - UNC: score of always predicting the observed rate (the Null model).
    - DSC: how much the score improves when the predictions are optimally
      recalibrated (isotonic regression), compared with UNC. Depends only on
      the ranking of the predictions.
    - MCB: how much worse the actual predictions score than their
      recalibrated version.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    s = SCORES[score]
    total = s(y, p)
    recalibrated = s(y, isotonic_recalibration(y, p))
    uncertainty = s(y, np.full_like(p, y.mean()))
    return pd.Series(
        {score: total, "MCB": total - recalibrated, "DSC": uncertainty - recalibrated, "UNC": uncertainty}
    )


def reliability_band(p, grid, n_resamples: int = 200, level: float = 0.9, seed: int = 0) -> pd.DataFrame:
    """Consistency band for a CORP reliability curve.

    Simulates outcomes from the predictions themselves (so the model is
    calibrated by construction), recomputes the isotonic curve each time, and
    returns pointwise quantiles at ``grid``. An observed curve outside the band
    is evidence of miscalibration beyond sampling noise.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(p, dtype=float)
    curves = np.empty((n_resamples, len(grid)))
    for k in range(n_resamples):
        y_sim = rng.random(p.size) < p
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y_sim)
        curves[k] = iso.predict(grid)
    tail = (1 - level) / 2
    return pd.DataFrame(
        {"p": grid, "lower": np.quantile(curves, tail, axis=0), "upper": np.quantile(curves, 1 - tail, axis=0)}
    )


# --- Continuous outcomes ---------------------------------------------------


def regression_metrics(y, pred, weight=None) -> pd.Series:
    """Error metrics for point predictions, optionally weighted."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, dtype=float)
    err = pred - y
    sse = np.average(err**2, weights=w)
    sst = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    return pd.Series(
        {
            "bias": np.average(err, weights=w),
            "MAE": np.average(np.abs(err), weights=w),
            "RMSE": np.sqrt(sse),
            "R²": 1 - sse / sst,
        }
    )


def tweedie_deviance(y, mu, power: float) -> np.ndarray:
    """Unit Tweedie deviance per observation, for 1 < power < 2.

    Reduces to (half of) the familiar losses at the boundaries: power 0 is
    squared error, 1 is Poisson, 2 is Gamma.
    """
    y, mu = np.asarray(y, dtype=float), np.asarray(mu, dtype=float)
    p = power
    return 2 * (
        np.power(np.maximum(y, 0), 2 - p) / ((1 - p) * (2 - p))
        - y * np.power(mu, 1 - p) / (1 - p)
        + np.power(mu, 2 - p) / (2 - p)
    )


def deviance_metrics(y, pred, null_pred, weight, power: float) -> pd.Series:
    """Mean Tweedie deviance and D² (share of the null model's deviance explained)."""
    dev = np.average(tweedie_deviance(y, pred, power), weights=weight)
    null_dev = np.average(tweedie_deviance(y, null_pred, power), weights=weight)
    return pd.Series({"mean deviance": dev, "D²": 1 - dev / null_dev})
