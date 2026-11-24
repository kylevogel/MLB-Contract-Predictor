import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
import boosted_tree as bt


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_no_contract_outcome_columns_used(group):
    assert not set(bt.FEATURES[group]) & bt.EXCLUDED
    X = bt.build_X(bt.load_group(group)["train"], group)
    assert list(X.columns) == bt.FEATURES[group]


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_test_classes_never_loaded(group):
    parts = bt.load_group(group)
    assert set(parts) == {"train", "val"}
    assert all(p["offseason"].max() <= 2023 for p in parts.values())


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_predictions_positive_and_right_length(group):
    parts = bt.load_group(group)
    X_tr = bt.build_X(parts["train"], group)
    X_va = bt.align_categories(X_tr, bt.build_X(parts["val"], group))
    for target in bt.TARGETS:
        y = parts["train"][target].to_numpy(float)
        model = bt.fit(X_tr, np.log(y) if target == "aav_nominal" else y, max_iter=20)
        pred = bt.predict(model, X_va, target)
        assert pred.shape == (len(X_va),)
        assert (pred > 0).all()


def test_score_years():
    m = bt.score_years([1, 2, 6], [1.2, 3.4, 8.0])
    assert m["years_mae"] == pytest.approx((0.2 + 1.4 + 2.0) / 3)
    assert m["exact_year"] == pytest.approx(1 / 3)
    assert m["years_bucket_accuracy"] == pytest.approx(2 / 3)  # 6 and 8 are both in the 6+ class


EXPECTED_BUCKET = {
    "hitters": {"C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
                "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH"},
    "pitchers": {"SP": "SP", "RP": "RP", "P": "RP"},
}


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_position_bucket_covers_every_position_in_the_data(group):
    df = pd.read_csv(bt.ROOT / "data/processed" / f"contracts_features_{group}.csv")
    firsts = set(df["position"].astype(str).str.split("/").str[0].str.strip())
    assert firsts == set(EXPECTED_BUCKET[group])  # a new position value must be mapped on purpose
    for pos in df["position"].unique():
        first = str(pos).split("/")[0].strip()
        assert bt.position_bucket(pos, group) == EXPECTED_BUCKET[group][first]


def test_position_bucket_rejects_unknown_hitter_position():
    with pytest.raises(ValueError):
        bt.position_bucket("XX", "hitters")


def test_results_csv_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "GRID", {"learning_rate": [0.1], "max_depth": [3], "max_iter": [20]})
    data = tmp_path / "data/processed"
    data.mkdir(parents=True)
    for g in bt.GROUPS:
        (data / f"contracts_features_{g}.csv").write_bytes(
            (pathlib.Path(__file__).parents[1] / "data/processed" / f"contracts_features_{g}.csv").read_bytes())
    bt.main()
    out = pd.read_csv(tmp_path / "results/boosted_tree_val_metrics.csv")
    assert list(out.columns) == [
        "model", "group", "n_train", "n_val",
        "within_5pct", "within_10pct", "mae_usd_m", "mae_log", "r2_log",
        "years_bucket_accuracy", "years_mae", "exact_year",
        "aav_learning_rate", "aav_max_depth", "aav_max_iter",
        "years_learning_rate", "years_max_depth", "years_max_iter"]
    assert list(out["group"]) == ["hitters", "pitchers"]
    assert (out["model"] == "boosted_tree").all()
