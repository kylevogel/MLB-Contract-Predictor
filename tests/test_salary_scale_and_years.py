import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from salary_scale import scale_for, to_share, from_share, ROOT
from metrics import years_bucket, bucket_from_years, years_class_report


def test_threshold_table_complete_and_sourced():
    t = pd.read_csv(ROOT / "data/raw/manual_cbt_threshold.csv")
    assert t["season"].tolist() == list(range(2015, 2027))
    assert t["source"].str.len().gt(10).all()
    assert t["cbt_threshold_m"].is_monotonic_increasing


def test_class_uses_next_season():
    assert scale_for([2014, 2021, 2022, 2025]).tolist() == [189, 230, 233, 244]
    with pytest.raises(ValueError):
        scale_for([2026])


def test_share_round_trip_on_real_data():
    for g in ("hitters", "pitchers"):
        df = pd.read_csv(ROOT / f"data/processed/contracts_features_{g}.csv")
        share = to_share(df["aav_nominal"], df["offseason"])
        assert (share > 0).all() and share.max() < 0.25
        assert np.allclose(from_share(share, df["offseason"]), df["aav_nominal"])


def test_years_bucket_and_rounding():
    assert years_bucket([1, 2, 3, 4, 5, 6, 15]).tolist() == [0, 1, 2, 3, 3, 4, 4]
    assert bucket_from_years([0.2, 1.49, 1.5, 2.5, 3.5, 5.49, 5.5]).tolist() == [0, 0, 1, 2, 3, 3, 4]


def test_years_class_report_balanced():
    true = [1, 1, 1, 1, 2, 6]
    always_one = years_class_report(true, [0] * 6)
    assert always_one["years_bucket_accuracy"] == pytest.approx(4 / 6)
    assert always_one["years_balanced_accuracy"] == pytest.approx(1 / 3)   # recall 1, 0, 0 over 3 classes
    perfect = years_class_report(true, years_bucket(true))
    assert perfect == {"years_bucket_accuracy": 1.0, "years_balanced_accuracy": 1.0}
