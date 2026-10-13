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


# --- Ranking and segmentation ----------------------------------------------


def exposure_bins(score, weight, n_bins: int = 10) -> np.ndarray:
    """Bin index per observation (0 = lowest scores), each bin holding the same total weight.

    An observation goes to the bin containing the midpoint of its share of the
    cumulative weight, so bins are as equal as individual weights allow.
    """
    score, w = np.asarray(score, dtype=float), np.asarray(weight, dtype=float)
    order = np.argsort(score, kind="mergesort")
    cum = np.cumsum(w[order])
    mid = (cum - w[order] / 2) / cum[-1]
    bins = np.empty(score.size, dtype=int)
    bins[order] = np.minimum((mid * n_bins).astype(int), n_bins - 1)
    return bins


def segment_table(y, preds: dict, weight, sort_by, n_bins: int = 10, level: float = 0.95) -> pd.DataFrame:
    """Weighted average outcome and predictions per bin of ``sort_by``: the data behind lift charts.

    - Lift chart: ``sort_by`` is a model's own prediction.
    - Double lift chart: ``sort_by`` is the ratio of two models' predictions.

    The interval for the actual average uses the standard error of a weighted
    mean, ``sqrt(Σ w²(y − ȳ)²) / Σ w``, with a normal approximation.
    """
    y, w = np.asarray(y, dtype=float), np.asarray(weight, dtype=float)
    sort_by = np.asarray(sort_by, dtype=float)
    bins = exposure_bins(sort_by, w, n_bins)
    z = norm.ppf(0.5 + level / 2)
    rows = []
    for b in range(n_bins):
        s = bins == b
        ws, ys = w[s], y[s]
        actual = np.average(ys, weights=ws)
        se = np.sqrt(np.sum(ws**2 * (ys - actual) ** 2)) / ws.sum()
        rows.append(
            {
                "bin": b + 1,
                "exposure": ws.sum(),
                "sort_min": sort_by[s].min(),
                "sort_max": sort_by[s].max(),
                "actual": actual,
                "lower": actual - z * se,
                "upper": actual + z * se,
                **{name: np.average(np.asarray(p, dtype=float)[s], weights=ws) for name, p in preds.items()},
            }
        )
    return pd.DataFrame(rows).set_index("bin")


def lorenz_curve(y, score, weight) -> pd.DataFrame:
    """Lorenz curve: share of total outcome against share of exposure, lowest scores first.

    Observations with equal scores are grouped, so ties give a straight
    segment instead of an arbitrary order (the Null model traces the diagonal).
    """
    y, w = np.asarray(y, dtype=float), np.asarray(weight, dtype=float)
    grouped = pd.DataFrame({"score": score, "yw": y * w, "w": w}).groupby("score", sort=True).sum()
    return pd.DataFrame(
        {
            "exposure_share": np.r_[0.0, grouped["w"].cumsum() / grouped["w"].sum()],
            "loss_share": np.r_[0.0, grouped["yw"].cumsum() / grouped["yw"].sum()],
        }
    )


def gini_index(y, score, weight) -> float:
    """Gini index: twice the area between the diagonal and the Lorenz curve.

    0 for a model that doesn't separate risks, larger when the low scores
    hold a smaller share of the outcome. Depends only on the ranking.
    """
    lc = lorenz_curve(y, score, weight)
    return float(1 - 2 * np.trapezoid(lc["loss_share"], lc["exposure_share"]))


# --- Model comparison --------------------------------------------------------


def paired_bootstrap(metric, y, preds: dict, weight=None, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Metric of each model on ``n_boot`` bootstrap resamples of the test set.

    Each resample draws as many observations as the test set has, with
    replacement, and every model is scored on the *same* resample, so the
    differences between columns are paired. ``metric`` is called as
    ``metric(y, pred)``, or ``metric(y, pred, weight)`` when a weight is given.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    preds = {name: np.asarray(p, dtype=float) for name, p in preds.items()}
    extra = () if weight is None else (np.asarray(weight, dtype=float),)
    rows = []
    for _ in range(n_boot):
        idx = rng.integers(y.size, size=y.size)
        rows.append({name: metric(y[idx], p[idx], *(e[idx] for e in extra)) for name, p in preds.items()})
    return pd.DataFrame(rows)


def bootstrap_summary(estimate: float, draws, level: float = 0.95) -> pd.Series:
    """Point estimate with the bootstrap's standard error, percentile interval and p-value.

    The two-sided p-value for 'no difference' is twice the share of resamples
    on the far side of 0 (roughly the smallest level at which the percentile
    interval would still include 0). Counting the observed sample as one more
    draw, (count + 1) / (B + 1), keeps it above 0: B resamples can't support a
    p-value much below 2 / B.
    """
    draws = np.asarray(draws, dtype=float)
    tail = (1 - level) / 2
    beyond = min(np.sum(draws <= 0), np.sum(draws >= 0))
    p = 2 * (beyond + 1) / (draws.size + 1)
    return pd.Series(
        {
            "difference": estimate,
            "std. error": draws.std(ddof=1),
            "lower": np.quantile(draws, tail),
            "upper": np.quantile(draws, 1 - tail),
            "p-value": min(p, 1.0),
        }
    )


def _z_test(estimate: float, se: float, level: float = 0.95) -> dict:
    """Normal-approximation interval and two-sided p-value for an estimate with standard error ``se``."""
    z = norm.ppf(0.5 + level / 2)
    return {
        "std. error": se,
        "lower": estimate - z * se,
        "upper": estimate + z * se,
        "p-value": 2 * norm.sf(abs(estimate / se)),
    }


def delong_test(y, score_a, score_b) -> pd.Series:
    """DeLong's test for the difference between two correlated AUCs (same cases).

    Each case gets a *placement value*: for a positive, the share of negatives
    it outscores; for a negative, the share of positives that outscore it
    (ties count 1/2). Either set of placements averages to the AUC, and their
    variances give the AUC's variance without resampling. Pairing enters
    through the per-case differences of placements between the two models.
    """
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()

    def placements(s):
        s = np.asarray(s, dtype=float)
        r = rankdata(s)
        below = r[y] - rankdata(s[y])  # negatives below each positive
        above = n_pos - (r[~y] - rankdata(s[~y]))  # positives above each negative
        return below / n_neg, above / n_pos

    pos_a, neg_a = placements(score_a)
    pos_b, neg_b = placements(score_b)
    diff = pos_a.mean() - pos_b.mean()
    var = np.var(pos_a - pos_b, ddof=1) / n_pos + np.var(neg_a - neg_b, ddof=1) / n_neg
    return pd.Series({"AUC A": pos_a.mean(), "AUC B": pos_b.mean(), "difference": diff, **_z_test(diff, np.sqrt(var))})


def diebold_mariano(loss_a, loss_b, weight=None) -> pd.Series:
    """Diebold–Mariano test: is the mean loss of model A different from model B's?

    Works on the per-observation *loss differential* d = loss_a − loss_b
    (positive when model B is better). It
    applies to any score that is an average over observations (squared error,
    log loss, deviance), not to AUC or Gini. For independent observations it is
    a paired z-test on d; the original version for forecasts over time replaces
    the standard error with one robust to autocorrelation.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    w = np.ones_like(d) if weight is None else np.asarray(weight, dtype=float)
    mean = np.average(d, weights=w)
    se = np.sqrt(np.sum(w**2 * (d - mean) ** 2)) / w.sum()
    return pd.Series({"difference": mean, **_z_test(mean, se)})


# --- Drift -------------------------------------------------------------------


def psi_table(reference, current, n_bins: int = 10, eps: float = 1e-4) -> pd.DataFrame:
    """Share of reference and current data per bin, and each bin's contribution to the PSI.

    Numeric features are cut at the reference deciles (fewer bins when values
    repeat); categorical features use one bin per category. Empty bins get the
    share ``eps`` so the logarithm stays finite;
    bins empty in both samples are dropped.
    """
    reference, current = pd.Series(reference), pd.Series(current)
    if isinstance(reference.dtype, pd.CategoricalDtype) or reference.dtype == object:
        ref_bins, cur_bins = reference.astype(str), current.astype(str)
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
        labels = [f"< {inner[0]:g}"] + [f"{lo:g} – {hi:g}" for lo, hi in zip(inner[:-1], inner[1:])] + [f"≥ {inner[-1]:g}"]
        to_label = lambda x: pd.Categorical.from_codes(np.searchsorted(inner, x, side="right"), labels)
        ref_bins, cur_bins = to_label(reference), to_label(current)
    t = pd.DataFrame({
        "reference": pd.Series(ref_bins).value_counts(normalize=True, sort=False),
        "current": pd.Series(cur_bins).value_counts(normalize=True, sort=False),
    }).fillna(0.0)
    t = t[t.sum(axis=1) > 0].clip(lower=eps)
    return t.assign(contribution=(t["current"] - t["reference"]) * np.log(t["current"] / t["reference"]))


def population_stability_index(reference, current, n_bins: int = 10) -> float:
    """PSI: Σ (current share − reference share) · ln(current share / reference share) over bins.

    It is the symmetric Kullback–Leibler divergence between the two binned
    distributions. Common rules of thumb: below 0.1 stable, 0.1–0.25 a
    moderate shift, above 0.25 a major shift.
    """
    return float(psi_table(reference, current, n_bins)["contribution"].sum())


def domain_classifier_auc(reference: pd.DataFrame, current: pd.DataFrame, n_splits: int = 3, seed: int = 0) -> float:
    """Cross-validated AUC of a classifier trained to tell current rows from reference rows.

    About 0.5 when the two samples come from the same distribution; the
    further above 0.5, the easier the samples are to tell apart. Unlike the
    PSI it sees all features at once, including changes in how they combine.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X = pd.concat([reference, current], ignore_index=True)
    is_current = np.r_[np.zeros(len(reference)), np.ones(len(current))]
    clf = HistGradientBoostingClassifier(categorical_features="from_dtype", max_iter=200, random_state=seed)
    folds = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    p = cross_val_predict(clf, X, is_current, cv=folds, method="predict_proba")[:, 1]
    return auc_rank(is_current, p)


def expected_auc(scores, prob) -> float:
    """AUC expected when each case is positive with probability ``prob`` and ranked by ``scores``.

    The ROC curve is built from *expected* counts: lowering the threshold past
    a case adds ``prob`` to the expected true positives and ``1 − prob`` to the
    expected false positives.
    """
    scores, prob = np.asarray(scores, dtype=float), np.asarray(prob, dtype=float)
    grouped = pd.DataFrame({"s": scores, "tp": prob, "fp": 1 - prob}).groupby("s", sort=True).sum()[::-1]
    tpr = np.r_[0.0, grouped["tp"].cumsum() / grouped["tp"].sum()]
    fpr = np.r_[0.0, grouped["fp"].cumsum() / grouped["fp"].sum()]
    return auc_trapezoid(fpr, tpr)


def cbpe(y_reference, p_reference, p_current) -> pd.Series:
    """Confidence-based performance estimation: binary metrics on unlabelled data.

    The predictions are first recalibrated on the labelled reference data
    (isotonic regression of outcome on prediction). The recalibrated
    probability of each current case is then taken as its chance of being
    positive, and every metric is replaced by its expected value. Valid as
    long as the outcome given the prediction behaves as in the reference data,
    which covariate shift preserves and concept drift breaks.
    """
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_reference, y_reference)
    p = np.asarray(p_current, dtype=float)
    q = np.clip(iso.predict(p), EPS, 1 - EPS)  # estimated chance of a claim
    pc = np.clip(p, EPS, 1 - EPS)
    return pd.Series({
        "positive rate": q.mean(),
        "AUC": expected_auc(p, q),
        "log loss": float(-np.mean(q * np.log(pc) + (1 - q) * np.log(1 - pc))),
        "Brier": float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2)),
    })
