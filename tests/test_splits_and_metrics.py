import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from splits import split_by_offseason, split_by_group, split_name
from metrics import within_pct, mae_log, r2_log, mae_usd_m, report

ROOT = pathlib.Path(__file__).parents[1]


def test_split_boundaries():
    assert [split_name(y) for y in (2014, 2021, 2022, 2023, 2024, 2025)] == ["train", "train", "val", "val", "test", "test"]
    with pytest.raises(ValueError):
        split_name(2026)


def test_real_data_partition_and_sizes():
    df = pd.read_csv(ROOT / "data/processed/contracts.csv")
    parts = split_by_offseason(df)
    assert {k: len(v) for k, v in parts.items()} == {"train": 1192, "val": 346, "test": 366}   # DATA_SPEC.md
    assert parts["train"].offseason.max() < parts["val"].offseason.min() <= parts["val"].offseason.max() < parts["test"].offseason.min()
    ids = [set(p.index) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    hit = split_by_group(df, "hitter")
    assert {k: len(v) for k, v in hit.items()} == {"train": 557, "val": 159, "test": 147}


def test_within_pct():
    assert within_pct([10, 10, 10], [10.5, 10.6, 9.4]) == pytest.approx(1 / 3)   # only 10.5 is within 5%
    assert within_pct([10], [10]) == 1.0


def test_log_metrics_and_perfect_fit():
    y = np.array([1.0, 2.0, 4.0, 8.0])
    assert mae_log(y, y) == 0 and r2_log(y, y) == 1 and mae_usd_m(y, y) == 0
    assert mae_log([4.0], [2.0]) == pytest.approx(np.log(2))
    assert r2_log(y, np.full(4, np.exp(np.log(y).mean()))) == pytest.approx(0)    # predicting the mean -> R2 of 0
    assert set(report(y, y)) == {"within_5pct", "within_10pct", "mae_usd_m", "mae_log", "r2_log"}


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        within_pct([1, 2], [1])
    with pytest.raises(ValueError):
        mae_log([0.0], [1.0])
