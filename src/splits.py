"""Time-based train / validation / test split by offseason class (DATA_SPEC.md, AGENTS.md).

Train 2014-2021, validate 2022-2023, test 2024-2025. The test classes are used once, after
every modeling choice is fixed. Splitting by class (never at random) keeps future contracts
out of training.
"""
import pandas as pd

SPLITS = {"train": (2014, 2021), "val": (2022, 2023), "test": (2024, 2025)}
POS_GROUPS = ("hitter", "pitcher")


def split_name(offseason: int) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= offseason <= hi:
            return name
    raise ValueError(f"offseason {offseason} is outside every split")


def split_by_offseason(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test'} frames that partition `df` (every row in exactly one)."""
    names = df["offseason"].map(split_name)
    parts = {name: df[names == name].copy() for name in SPLITS}
    assert sum(len(p) for p in parts.values()) == len(df)
    return parts


def split_by_group(df: pd.DataFrame, pos_group: str) -> dict[str, pd.DataFrame]:
    """Hitters and pitchers are modeled separately; return the three splits for one group."""
    if pos_group not in POS_GROUPS:
        raise ValueError(f"pos_group must be one of {POS_GROUPS}")
    return split_by_offseason(df[df["pos_group"] == pos_group])
