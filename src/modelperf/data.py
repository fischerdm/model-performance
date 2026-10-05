"""Datasets used throughout the project.

Main dataset: French motor third-party liability (freMTPL2), combining two
OpenML tables:

- freMTPL2freq (id 41214): one row per policy with risk features and exposure.
- freMTPL2sev (id 41215): one row per claim with its amount.

From them we derive two targets:

- ``HasClaim`` (binary): did the policy have at least one claim?
- ``PurePremium`` (continuous, weight ``Exposure``): claim cost per year of
  exposure. Mostly zeros with a very long right tail, as insurance targets are.

Supporting dataset: California Housing (median house value per district), a
continuous target with a strong signal, used to show error metrics behaving
'normally' before turning to pure premium.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_HOME = PROJECT_ROOT / "data" / "raw"

FREQ_ID = 41214
SEV_ID = 41215

# Total claim cost per policy is capped to keep a handful of catastrophic
# claims (up to 4M EUR) from dominating every metric.
CLAIM_AMOUNT_CAP = 200_000

CATEGORICAL = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
FEATURES = CATEGORICAL + NUMERIC

RANDOM_STATE = 42
TEST_SIZE = 0.25


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


@lru_cache(maxsize=1)
def load_policies() -> pd.DataFrame:
    """Return one row per policy with features and both targets."""
    freq = fetch_openml(data_id=FREQ_ID, as_frame=True, data_home=DATA_HOME).frame
    sev = fetch_openml(data_id=SEV_ID, as_frame=True, data_home=DATA_HOME).frame

    freq["IDpol"] = freq["IDpol"].astype(int)
    freq["VehGas"] = freq["VehGas"].str.strip("'")
    for col in CATEGORICAL:
        freq[col] = freq[col].astype("category")

    claims = sev.groupby("IDpol")["ClaimAmount"].agg(ClaimAmount="sum", ClaimNb="count")
    df = freq.drop(columns="ClaimNb").merge(claims, on="IDpol", how="left")

    # The claim count is taken from the severity table so that both targets
    # are consistent: a policy 'has a claim' iff it has a recorded amount.
    df["ClaimNb"] = df["ClaimNb"].fillna(0).astype(int)
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0).clip(upper=CLAIM_AMOUNT_CAP)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)

    df["HasClaim"] = (df["ClaimNb"] > 0).astype(int)
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    return df.set_index("IDpol")


@lru_cache(maxsize=1)
def train_test() -> Split:
    """Fixed, stratified train/test split of all policies."""
    df = load_policies()
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["HasClaim"]
    )
    return Split(train=train, test=test)


@lru_cache(maxsize=1)
def housing_split() -> Split:
    """California Housing; target ``MedHouseVal`` is in units of 100,000 USD."""
    df = fetch_california_housing(as_frame=True, data_home=DATA_HOME).frame
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return Split(train=train, test=test)
