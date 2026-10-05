"""Check the from-scratch metrics against scikit-learn on synthetic data."""

import numpy as np
import pytest
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
