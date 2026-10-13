"""Simulated production data for the drift page.

The test set is split in half:

- the **reference** half plays the labelled data the model was validated on;
- the **pool** half supplies the policies of simulated monthly production
  batches, so no policy appears in both.

The split is stratified by claim, fuel type, area and age band (see
``reference_and_pool``).

Each scenario turns a drift ``strength`` between 0 (no drift) and 1 (full
drift) into a batch. Strength grows linearly over the months, so month 1 looks
like the reference and the last month shows the full effect.

- ``covariate``: the portfolio gets younger. Young drivers are sampled more
  often, which shifts the features; claim risk *given* the features is unchanged.
- ``concept``: the features are unchanged, but claims rise by half in the
  rural areas A–C, which the model has never seen.
- ``benign``: regular-petrol cars replace diesels, a feature the models
  hardly use and that barely affects claim risk.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import RANDOM_STATE, Split, train_test

SCENARIOS = {"covariate": "Covariate shift", "concept": "Concept drift", "benign": "Benign drift"}
N_MONTHS = 12
BATCH_SIZE = 20_000

# Full-strength settings of each scenario.
YOUNG_TILT = 1.2  # log sampling weight per 15 years below age 45
RURAL_AREAS = ["A", "B", "C"]
RURAL_CLAIM_INCREASE = 0.5  # claim rate in rural areas rises by 50%
DIESEL_SHARE = 0.15  # diesel share of the portfolio at full strength


@lru_cache(maxsize=1)
def reference_and_pool() -> Split:
    """Test set split in half: ``train`` is the reference, ``test`` the production pool.

    The split is stratified by claim, fuel type, area and age band: the
    segments the scenarios change then have the same claim rate in both halves,
    so differences between reference and production come from the simulated
    drift, not from the luck of the split.
    """
    test = train_test().test
    age_band = pd.cut(test["DrivAge"], [0, 30, 40, 50, 60, np.inf])
    strata = test[["HasClaim", "VehGas", "Area"]].astype(str).agg("-".join, axis=1) + age_band.astype(str)
    reference, pool = train_test_split(test, test_size=0.5, random_state=RANDOM_STATE, stratify=strata)
    return Split(train=reference, test=pool)


def _sample(pool: pd.DataFrame, log_weight, size: int, rng) -> pd.DataFrame:
    """Draw ``size`` distinct policies with probability proportional to ``exp(log_weight)``."""
    w = np.exp(np.asarray(log_weight, dtype=float))
    idx = rng.choice(len(pool), size=size, replace=False, p=w / w.sum())
    return pool.iloc[idx].copy()


def simulate_batch(scenario: str, strength: float, size: int = BATCH_SIZE, seed: int = 0) -> pd.DataFrame:
    """One production batch of ``size`` policies under ``scenario`` at the given ``strength``."""
    rng = np.random.default_rng(seed)
    pool = reference_and_pool().test
    zero = np.zeros(len(pool))

    if scenario == "covariate":
        younger = -(pool["DrivAge"].to_numpy() - 45) / 15
        return _sample(pool, strength * YOUNG_TILT * younger, size, rng)

    if scenario == "benign":
        diesel = (pool["VehGas"] == "Diesel").to_numpy()
        share = diesel.mean()
        target = share + strength * (DIESEL_SHARE - share)
        # Reweight diesel against regular cars until the batch has the target share.
        tilt = np.log(target / (1 - target)) - np.log(share / (1 - share))
        return _sample(pool, np.where(diesel, tilt, 0.0), size, rng)

    if scenario == "concept":
        batch = _sample(pool, zero, size, rng)
        rural = batch["Area"].isin(RURAL_AREAS).to_numpy()
        rate = batch.loc[rural, "HasClaim"].mean()
        # Turn no-claim policies into claims at a constant rate: every rural
        # policy gains the same extra claim probability.
        flip = strength * RURAL_CLAIM_INCREASE * rate / (1 - rate)
        new_claim = rural & (batch["HasClaim"].to_numpy() == 0) & (rng.random(size) < flip)
        batch.loc[new_claim, "HasClaim"] = 1
        return batch

    raise ValueError(f"unknown scenario {scenario!r}; choose from {list(SCENARIOS)}")


def simulate_months(scenario: str, n_months: int = N_MONTHS, size: int = BATCH_SIZE, seed: int = 0) -> pd.DataFrame:
    """Monthly batches with drift growing linearly from none (month 1) to full (last month)."""
    batches = [
        simulate_batch(scenario, m / (n_months - 1), size, seed=seed + m).assign(month=m + 1)
        for m in range(n_months)
    ]
    return pd.concat(batches)
