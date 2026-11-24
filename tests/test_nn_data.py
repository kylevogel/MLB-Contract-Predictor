import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
from nn_data import load_group, prepare, years_bucket, position_bucket, ROOT

LEAKY = ["at_league_minimum", "aav_adjusted", "deferred_amount_m", "has_deferral", "offseason",
         "service_years_at_signing", "has_opt_out", "has_incentives"]


def test_years_bucket_edges():
    assert years_bucket([1, 2, 3, 4, 5, 6, 15]).tolist() == [0, 1, 2, 3, 3, 4, 4]
    with pytest.raises(ValueError):
        years_bucket([0])


def test_position_bucket():
    assert position_bucket("1B/DH", "hitters") == "IF"
    assert position_bucket("C", "hitters") == "C"
    assert position_bucket("OF", "hitters") == "OF"
    assert position_bucket("SP/SP2", "pitchers") == "SP"
    assert position_bucket("RP/CL", "pitchers") == "RP"
    assert position_bucket("P", "pitchers") == "RP"
    with pytest.raises(ValueError):
        position_bucket("XX", "hitters")


@pytest.mark.parametrize("group,n_train,n_val", [("hitters", 557, 159), ("pitchers", 632, 185)])
def test_real_sizes_no_nans_no_leaky_columns(group, n_train, n_val):
    d = load_group(group)
    assert d.X_train.shape == (n_train, len(d.feature_names))
    assert d.X_val.shape == (n_val, len(d.feature_names))
    assert not np.isnan(d.X_train).any() and not np.isnan(d.X_val).any()
    assert not set(LEAKY) & set(d.feature_names)
    assert d.meta_val["offseason"].between(2022, 2023).all()
    assert d.meta_train["offseason"].between(2014, 2021).all()
    # Train columns are standardized (constant columns stay at 0).
    assert np.allclose(d.X_train.mean(axis=0), 0, atol=1e-5)


def test_no_test_rows_returned():
    df = pd.read_csv(ROOT / "data/processed/contracts_features_hitters.csv")
    d = prepare(df, "hitters")
    assert len(d.meta_train) + len(d.meta_val) == (df["offseason"] <= 2023).sum()


def test_scaler_uses_train_only():
    df = pd.read_csv(ROOT / "data/processed/contracts_features_pitchers.csv")
    base = prepare(df, "pitchers")
    changed = df.copy()
    val_idx = changed.index[changed["offseason"].between(2022, 2023)]
    changed.loc[val_idx, "war_3yr"] = 99.0  # wildly change every validation row
    changed.loc[val_idx[0], "era_plus_3yr"] = np.nan
    other = prepare(changed, "pitchers")
    assert np.array_equal(base.mean, other.mean) and np.array_equal(base.std, other.std)
    assert np.array_equal(base.X_train, other.X_train)
    assert not np.isnan(other.X_val).any()  # val NaN filled with the train median
