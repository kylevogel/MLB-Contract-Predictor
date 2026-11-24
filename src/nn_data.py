"""Neural-network input prep (issue #20).

Turns data/processed/contracts_features_{hitters,pitchers}.csv into standardized arrays for
the network. Hitters and pitchers are prepared separately (D-09). Only the train and validation
classes are returned: the test classes (2024-2025) are used once, at the very end.

Feature list agreed for all three models in DECISIONS.md D-18 (is_resign is not used: it is True
for every extension and, for free agents, says where the player signed, which is part of the outcome).

Targets: log AAV share of the luxury-tax threshold (D-19; `y_log_share_*`, the one the network fits),
log nominal AAV (`y_log_aav_*`, kept for reference) and the length class (`y_bucket_*`).

Scaling (mean / std) and the median used to fill missing stats come from train rows only,
so nothing about the validation contracts leaks into the inputs.

Usage:
    from nn_data import load_group
    data = load_group("hitters")
    data.X_train, data.y_log_share_train, data.y_bucket_train, data.feature_names
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import years_bucket
from salary_scale import to_share
from splits import split_by_group

ROOT = Path(__file__).parents[1]
GROUPS = {"hitters": "hitter", "pitchers": "pitcher"}

CONTRACT_TYPES = ["free_agent", "extension_pre_arb", "extension_arb", "extension_veteran"]
COMMON = ["age_at_signing", "seasons_in_window",
          "won_mvp", "won_cy_young", "won_gold_glove", "won_silver_slugger"]
PREV = ["has_prev_contract", "prev_contract_years", "log_prev_contract_aav", "years_since_prev_contract"]
STATS = {"hitters": ["war_3yr", "r_3yr", "ba_3yr", "ops_plus_3yr"],
         "pitchers": ["war_3yr", "era_plus_3yr", "fip_3yr", "whip_3yr"]}
BUCKETS = {"hitters": ["C", "IF", "OF", "DH"], "pitchers": ["SP", "RP"]}
HITTER_BUCKET = {"C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
                 "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH"}

# Contract length classes (metrics.years_bucket): 1, 2, 3, 4-5, 6+ years -> 0..4. Midpoints turn a class back into years.
YEAR_BUCKET_LABELS = ["1", "2", "3", "4-5", "6+"]
YEAR_BUCKET_MIDPOINTS = np.array([1.0, 2.0, 3.0, 4.5, 7.0])


def position_bucket(position: str, group: str) -> str:
    """First listed position, grouped: hitters C/IF/OF/DH, pitchers SP/RP (generic 'P' -> RP)."""
    first = str(position).split("/")[0].strip()
    if group == "hitters":
        if first not in HITTER_BUCKET:
            raise ValueError(f"unknown hitter position {position!r}")
        return HITTER_BUCKET[first]
    return "SP" if first == "SP" else "RP"


def build_frame(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Unscaled feature frame, in a fixed column order. Missing stats are left as NaN here."""
    out = pd.DataFrame(index=df.index)
    for c in COMMON:
        out[c] = df[c].astype(float)
    for t in CONTRACT_TYPES:
        out[f"type_{t}"] = (df["contract_type"] == t).astype(float)
    for t in CONTRACT_TYPES:  # previous contract's type; all zeros when there is none
        out[f"prev_type_{t}"] = (df["prev_contract_type"] == t).astype(float)
    # Previous contract: has_prev_contract is the mask; the rest are 0 when there is none (D-10).
    has_prev = df["has_prev_contract"].astype(bool)
    out["has_prev_contract"] = has_prev.astype(float)
    out["prev_contract_years"] = df["prev_contract_years"].where(has_prev, 0).fillna(0).astype(float)
    out["log_prev_contract_aav"] = np.log1p(df["prev_contract_aav"].where(has_prev, 0).fillna(0).astype(float))
    out["years_since_prev_contract"] = df["years_since_prev_contract"].where(has_prev, 0).fillna(0).astype(float)
    for c in STATS[group]:
        out[c] = df[c].astype(float)
    buckets = df["position"].map(lambda p: position_bucket(p, group))
    for b in BUCKETS[group]:
        out[f"pos_{b}"] = (buckets == b).astype(float)
    return out


@dataclass
class GroupData:
    feature_names: list[str]
    X_train: np.ndarray
    X_val: np.ndarray
    y_log_aav_train: np.ndarray
    y_log_aav_val: np.ndarray
    y_log_share_train: np.ndarray  # log(AAV / CBT threshold), the fitted target (D-19)
    y_log_share_val: np.ndarray
    y_bucket_train: np.ndarray
    y_bucket_val: np.ndarray
    meta_train: pd.DataFrame  # player, offseason, actual targets: for writing predictions
    meta_val: pd.DataFrame
    mean: np.ndarray
    std: np.ndarray


def prepare(df: pd.DataFrame, group: str) -> GroupData:
    """Split one group's contracts by offseason, then fill and scale with train statistics only."""
    if group not in GROUPS:
        raise ValueError(f"group must be one of {list(GROUPS)}")
    parts = split_by_group(df, GROUPS[group])
    train, val = parts["train"], parts["val"]  # parts["test"] is deliberately never used
    f_train, f_val = build_frame(train, group), build_frame(val, group)

    medians = f_train.median()
    f_train, f_val = f_train.fillna(medians), f_val.fillna(medians)
    mean = f_train.mean().to_numpy(copy=True)
    std = f_train.std(ddof=0).to_numpy(copy=True)
    std[std == 0] = 1.0  # constant columns (e.g. won_mvp for pitchers) stay at 0 after centering

    def scale(f: pd.DataFrame) -> np.ndarray:
        return ((f.to_numpy() - mean) / std).astype(np.float32)

    meta_cols = ["player_name", "mlb_id", "offseason", "contract_type", "years_new", "aav_nominal"]
    return GroupData(
        feature_names=list(f_train.columns),
        X_train=scale(f_train), X_val=scale(f_val),
        y_log_aav_train=np.log(train["aav_nominal"].to_numpy(dtype=np.float32)),
        y_log_aav_val=np.log(val["aav_nominal"].to_numpy(dtype=np.float32)),
        y_log_share_train=np.log(to_share(train["aav_nominal"], train["offseason"])).astype(np.float32),
        y_log_share_val=np.log(to_share(val["aav_nominal"], val["offseason"])).astype(np.float32),
        y_bucket_train=years_bucket(train["years_new"]),
        y_bucket_val=years_bucket(val["years_new"]),
        meta_train=train[meta_cols].reset_index(drop=True),
        meta_val=val[meta_cols].reset_index(drop=True),
        mean=mean, std=std,
    )


def load_group(group: str) -> GroupData:
    """Read the feature CSV for 'hitters' or 'pitchers' and prepare train / validation arrays."""
    df = pd.read_csv(ROOT / f"data/processed/contracts_features_{group}.csv")
    return prepare(df, group)
