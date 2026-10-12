"""Check the from-scratch metrics against scikit-learn on synthetic data."""

import numpy as np
import pytest
from scipy import stats
from scipy.special import expit, logit
from sklearn import metrics as skm

from modelperf import metrics as mp

rng = np.random.default_rng(0)
N = 2_000


@pytest.fixture
def binary():
    y = rng.binomial(1, 0.2, N)
    # Rounded scores create ties, which ROC and AUC must handle correctly.
    p = np.round(np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.2, N), 0, 1), 2)
    return y, p


@pytest.fixture
def continuous():
    y = rng.gamma(2.0, 50.0, N)
    pred = y * rng.lognormal(0, 0.3, N)
    w = rng.uniform(0.1, 1.0, N)
    return y, pred, w


def test_confusion_counts(binary):
    y, p = binary
    c = mp.confusion_counts(y, p, 0.3)
    tn, fp, fn, tp = skm.confusion_matrix(y, p >= 0.3).ravel()
    assert (c["TN"], c["FP"], c["FN"], c["TP"]) == (tn, fp, fn, tp)


def test_threshold_metrics(binary):
    y, p = binary
    m = mp.threshold_metrics(y, p, 0.3)
    pred = p >= 0.3
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, pred))
    assert m["precision"] == pytest.approx(skm.precision_score(y, pred))
    assert m["recall (TPR)"] == pytest.approx(skm.recall_score(y, pred))
    assert m["F1"] == pytest.approx(skm.f1_score(y, pred))
    assert m["balanced accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, pred))


def test_roc_points(binary):
    y, p = binary
    roc = mp.roc_points(y, p)
    fpr, tpr, thr = skm.roc_curve(y, p, drop_intermediate=False)
    np.testing.assert_allclose(roc["fpr"], fpr)
    np.testing.assert_allclose(roc["tpr"], tpr)
    np.testing.assert_allclose(roc["threshold"][1:], thr[1:])


def test_auc_two_ways(binary):
    y, p = binary
    roc = mp.roc_points(y, p)
    expected = skm.roc_auc_score(y, p)
    assert mp.auc_trapezoid(roc["fpr"], roc["tpr"]) == pytest.approx(expected)
    assert mp.auc_rank(y, p) == pytest.approx(expected)


def test_pr_points(binary):
    y, p = binary
    pr = mp.pr_points(y, p)
    precision, recall, _ = skm.precision_recall_curve(y, p, drop_intermediate=False)
    # scikit-learn lists points with recall decreasing and appends (recall 0, precision 1).
    np.testing.assert_allclose(pr["recall"], recall[:-1][::-1])
    np.testing.assert_allclose(pr["precision"], precision[:-1][::-1])


def test_average_precision(binary):
    y, p = binary
    assert mp.average_precision(y, p) == pytest.approx(skm.average_precision_score(y, p))


def test_top_share_metrics():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    s = np.arange(10, 0, -1)  # scores already in decreasing order
    m = mp.top_share_metrics(y, s, 0.2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["lift"] == pytest.approx(2.5)


@pytest.fixture
def calibrated():
    """Outcomes drawn from the predicted probabilities: calibrated by construction."""
    p = expit(rng.normal(-2.0, 1.0, 50_000))
    y = (rng.random(p.size) < p).astype(int)
    return y, p


def test_scores(binary):
    y, p = binary
    assert mp.brier_score(y, p) == pytest.approx(skm.brier_score_loss(y, p))
    assert mp.log_loss(y, p) == pytest.approx(skm.log_loss(y, np.clip(p, mp.EPS, 1 - mp.EPS)))


@pytest.mark.parametrize("score", ["Brier", "log loss"])
def test_murphy_decomposition(binary, score):
    y, p = binary
    d = mp.murphy_decomposition(y, p, score)
    assert d[score] == pytest.approx(d["MCB"] - d["DSC"] + d["UNC"])
    assert d["MCB"] >= 0 and d["DSC"] >= 0
    # DSC depends only on the ranking: a monotone transformation keeps it.
    assert mp.murphy_decomposition(y, p**3, score)["DSC"] == pytest.approx(d["DSC"])


def test_calibration_bins(binary):
    y, p = binary
    t = mp.calibration_bins(y, p, n_bins=10, strategy="uniform")
    assert t["n"].sum() == len(y)
    assert np.average(t["observed_rate"], weights=t["n"]) == pytest.approx(y.mean())
    assert (t["lower"] <= t["observed_rate"]).all() and (t["observed_rate"] <= t["upper"]).all()


def test_intercept_slope(calibrated):
    y, p = calibrated
    s = mp.calibration_intercept_slope(y, p)
    assert s["calibration intercept"] == pytest.approx(0, abs=0.05)
    assert s["calibration slope"] == pytest.approx(1, abs=0.05)
    # Doubling the logits makes predictions too extreme: the slope halves.
    too_extreme = expit(2 * logit(p))
    assert mp.calibration_intercept_slope(y, too_extreme)["calibration slope"] == pytest.approx(0.5, abs=0.03)


def test_regression_metrics(continuous):
    y, pred, w = continuous
    m = mp.regression_metrics(y, pred, w)
    assert m["MAE"] == pytest.approx(skm.mean_absolute_error(y, pred, sample_weight=w))
    assert m["RMSE"] == pytest.approx(skm.root_mean_squared_error(y, pred, sample_weight=w))
    assert m["R²"] == pytest.approx(skm.r2_score(y, pred, sample_weight=w))


def test_tweedie_deviance(continuous):
    y, pred, w = continuous
    y = np.where(rng.random(N) < 0.5, 0.0, y)  # Tweedie allows exact zeros
    dev = np.average(mp.tweedie_deviance(y, pred, 1.9), weights=w)
    assert dev == pytest.approx(skm.mean_tweedie_deviance(y, pred, power=1.9, sample_weight=w))
    d2 = mp.deviance_metrics(y, pred, np.full(N, np.average(y, weights=w)), w, 1.9)["D²"]
    assert d2 == pytest.approx(skm.d2_tweedie_score(y, pred, power=1.9, sample_weight=w))


def test_exposure_bins(continuous):
    _, pred, w = continuous
    assert np.bincount(mp.exposure_bins(pred, np.ones(N), 10)).tolist() == [N // 10] * 10
    totals = np.bincount(mp.exposure_bins(pred, w, 10), weights=w)
    np.testing.assert_allclose(totals, w.sum() / 10, atol=w.max())
    # Bins follow the score: every score in bin k is below every score in bin k + 1.
    bins = mp.exposure_bins(pred, w, 10)
    assert all(pred[bins == k].max() <= pred[bins == k + 1].min() for k in range(9))


def test_segment_table(continuous):
    y, pred, w = continuous
    t = mp.segment_table(y, {"model": pred}, w, sort_by=pred)
    assert t["exposure"].sum() == pytest.approx(w.sum())
    # Bin averages, weighted by bin exposure, recover the overall averages.
    assert np.average(t["actual"], weights=t["exposure"]) == pytest.approx(np.average(y, weights=w))
    assert np.average(t["model"], weights=t["exposure"]) == pytest.approx(np.average(pred, weights=w))


def test_gini_index_binary(binary):
    """With a 0/1 outcome and unit weights, Gini index = (1 − positive rate) · (2·AUC − 1)."""
    y, p = binary
    expected = (1 - y.mean()) * (2 * skm.roc_auc_score(y, p) - 1)
    assert mp.gini_index(y, p, np.ones(N)) == pytest.approx(expected)


def test_gini_index_properties(continuous):
    y, pred, w = continuous
    g = mp.gini_index(y, pred, w)
    assert mp.gini_index(y, 3 * pred + 10, w) == pytest.approx(g)  # ranking only
    assert mp.gini_index(y, np.full(N, 5.0), w) == pytest.approx(0)  # constant: diagonal
    assert mp.gini_index(y, -pred, w) == pytest.approx(-g)  # reversed ranking
    assert mp.gini_index(y, y, w) > g  # sorting by the outcome itself is the best possible


def test_delong_matches_bootstrap(binary):
    y, p = binary
    other = np.clip(p + rng.normal(0, 0.1, N), 0, 1)  # a correlated, noisier competitor
    d = mp.delong_test(y, p, other)
    assert d["AUC A"] == pytest.approx(skm.roc_auc_score(y, p))
    assert d["AUC B"] == pytest.approx(skm.roc_auc_score(y, other))
    boot = mp.paired_bootstrap(mp.auc_rank, y, {"A": p, "B": other}, n_boot=500)
    assert d["std. error"] == pytest.approx((boot["A"] - boot["B"]).std(), rel=0.15)


def test_diebold_mariano_is_paired_test(continuous):
    y, pred, _ = continuous
    loss_a, loss_b = (pred - y) ** 2, (1.1 * pred - y) ** 2
    dm = mp.diebold_mariano(loss_a, loss_b)
    t = stats.ttest_rel(loss_a, loss_b)
    assert dm["difference"] == pytest.approx(np.mean(loss_a - loss_b))
    assert dm["p-value"] == pytest.approx(t.pvalue, rel=0.05)


def test_bootstrap_summary():
    draws = np.linspace(-1, 3, 1001)  # 250 of 1001 draws at or below 0
    s = mp.bootstrap_summary(1.0, draws, level=0.9)
    assert s["difference"] == 1.0
    assert (s["lower"], s["upper"]) == pytest.approx((-0.8, 2.8))
    assert s["p-value"] == pytest.approx(2 * 252 / 1002)
    assert mp.bootstrap_summary(1.0, np.ones(999))["p-value"] == pytest.approx(0.002)
