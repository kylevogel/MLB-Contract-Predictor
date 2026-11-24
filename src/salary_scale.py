"""League salary scale for the AAV target (DECISIONS.md D-19).

Salaries drift upward over time (the 2022 CBA raised the luxury-tax threshold by $20M in one year),
so models fit AAV as a share of that season's Competitive Balance Tax threshold and convert back to
nominal $M for scoring. Offseason class Y uses the threshold of season Y+1, the first season a
class-Y contract pays. Thresholds are hand-kept, with sources, in data/raw/manual_cbt_threshold.csv.

Usage:
    from salary_scale import to_share, from_share
    share = to_share(df["aav_nominal"], df["offseason"])       # fit log(share)
    aav = from_share(predicted_share, df["offseason"])         # back to $M
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
_TABLE = pd.read_csv(ROOT / "data/raw/manual_cbt_threshold.csv").set_index("season")["cbt_threshold_m"]


def scale_for(offseason) -> np.ndarray:
    """CBT threshold in $M for each offseason class (season = class + 1)."""
    seasons = np.asarray(offseason, dtype=int) + 1
    missing = sorted(set(seasons.ravel()) - set(_TABLE.index))
    if missing:
        raise ValueError(f"no CBT threshold for season(s) {missing}")
    return _TABLE.loc[seasons.ravel()].to_numpy(dtype=float).reshape(seasons.shape)


def to_share(aav, offseason) -> np.ndarray:
    """AAV in $M -> share of the class's CBT threshold."""
    return np.asarray(aav, dtype=float) / scale_for(offseason)


def from_share(share, offseason) -> np.ndarray:
    """Share of the CBT threshold -> AAV in $M."""
    return np.asarray(share, dtype=float) * scale_for(offseason)
