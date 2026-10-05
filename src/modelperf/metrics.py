"""Performance metrics, implemented from their definitions.

scikit-learn provides most of these already. They are re-implemented here so
the documentation can show exactly what each metric computes; ``tests/``
checks every function against the scikit-learn reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
