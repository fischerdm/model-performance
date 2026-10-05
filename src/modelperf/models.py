"""Fit (or load cached) baseline and challenger models for each target.

Every target gets the same three models:

- ``Null``: predicts the training average for everyone (the 'no model' reference).
- ``GLM``: a generalized linear model with a loss suited to the target.
- ``GBM``: histogram gradient boosting.

| Target        | GLM                         | GBM loss      | Weight   |
|---------------|-----------------------------|---------------|----------|
| HasClaim      | logistic regression         | log loss      | -        |
| PurePremium   | Tweedie (power 1.9, log)    | Poisson       | Exposure |
| MedHouseVal   | linear regression           | squared error | -        |

Fitting takes about a minute, so fitted models are cached in ``artifacts/``.
Bump ``MODEL_VERSION`` whenever data preparation or model specs change.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, TweedieRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from .data import CATEGORICAL, FEATURES, PROJECT_ROOT, RANDOM_STATE, housing_split, train_test

MODEL_VERSION = "1"
ARTIFACTS = PROJECT_ROOT / "artifacts"

BINARY_FEATURES = FEATURES + ["Exposure"]
TWEEDIE_POWER = 1.9

log_scaled = make_pipeline(FunctionTransformer(np.log), StandardScaler())


def _glm_preprocessor(with_exposure: bool) -> ColumnTransformer:
    """One-hot categoricals, binned ages, log-scaled skewed numerics."""
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL + ["VehPower"]),
        (
            "binned",
            KBinsDiscretizer(n_bins=10, encode="onehot", quantile_method="averaged_inverted_cdf"),
            ["VehAge", "DrivAge"],
        ),
        ("log", log_scaled, ["Density", "BonusMalus"]),
    ]
    if with_exposure:
        transformers.append(("exposure", log_scaled, ["Exposure"]))
    return ColumnTransformer(transformers)


def _fit_binary() -> dict:
    train = train_test().train
    X, y = train[BINARY_FEATURES], train["HasClaim"]
    glm = make_pipeline(_glm_preprocessor(with_exposure=True), LogisticRegression(max_iter=2000))
    gbm = HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=1000,
        min_samples_leaf=200,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    return {
        "Null": DummyClassifier(strategy="prior").fit(X, y),
        "GLM": glm.fit(X, y),
        "GBM": gbm.fit(X, y),
    }


def _fit_pure_premium() -> dict:
    train = train_test().train
    X, y, w = train[FEATURES], train["PurePremium"], train["Exposure"]
    glm = make_pipeline(
        _glm_preprocessor(with_exposure=False),
        TweedieRegressor(power=TWEEDIE_POWER, link="log", alpha=1e-2, max_iter=1000),
    )
    # scikit-learn's GBM has no Tweedie loss; Poisson deviance is the usual
    # stand-in and, like Tweedie, reproduces the (weighted) mean on training data.
    gbm = HistGradientBoostingRegressor(
        loss="poisson",
        categorical_features="from_dtype",
        learning_rate=0.03,
        max_iter=1000,
        min_samples_leaf=2000,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    # Pure premium is a rate per year, so each policy is weighted by its exposure.
    return {
        "Null": DummyRegressor(strategy="mean").fit(X, y, sample_weight=w),
        "GLM": glm.fit(X, y, tweedieregressor__sample_weight=w),
        "GBM": gbm.fit(X, y, sample_weight=w),
    }


def _fit_housing() -> dict:
    train = housing_split().train
    X, y = train.drop(columns="MedHouseVal"), train["MedHouseVal"]
    glm = make_pipeline(StandardScaler(), LinearRegression())
    gbm = HistGradientBoostingRegressor(max_iter=500, early_stopping=True, random_state=RANDOM_STATE)
    return {
        "Null": DummyRegressor(strategy="mean").fit(X, y),
        "GLM": glm.fit(X, y),
        "GBM": gbm.fit(X, y),
    }


@lru_cache(maxsize=1)
def load_models(refit: bool = False) -> dict:
    """Return ``{"binary": {...}, "pure_premium": {...}, "housing": {...}}``."""
    path = ARTIFACTS / f"models_v{MODEL_VERSION}.joblib"
    if path.exists() and not refit:
        return joblib.load(path)
    models = {
        "binary": _fit_binary(),
        "pure_premium": _fit_pure_premium(),
        "housing": _fit_housing(),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(models, path)
    return models


def binary_predictions() -> pd.DataFrame:
    """Test set: outcome ``y`` and predicted claim probability per model."""
    test = train_test().test
    preds = {
        name: model.predict_proba(test[BINARY_FEATURES])[:, 1]
        for name, model in load_models()["binary"].items()
    }
    return pd.DataFrame({"y": test["HasClaim"], **preds}, index=test.index)


def pure_premium_predictions() -> pd.DataFrame:
    """Test set: pure premium ``y``, exposure weight ``w`` and predictions per model."""
    test = train_test().test
    preds = {name: model.predict(test[FEATURES]) for name, model in load_models()["pure_premium"].items()}
    return pd.DataFrame({"y": test["PurePremium"], "w": test["Exposure"], **preds}, index=test.index)


def housing_predictions() -> pd.DataFrame:
    """Test set: median house value ``y`` (100k USD) and predictions per model."""
    test = housing_split().test
    X = test.drop(columns="MedHouseVal")
    preds = {name: model.predict(X) for name, model in load_models()["housing"].items()}
    return pd.DataFrame({"y": test["MedHouseVal"], **preds}, index=test.index)
