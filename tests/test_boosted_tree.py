import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
import boosted_tree as bt
from salary_scale import to_share

DATA = pathlib.Path(__file__).parents[1] / "data/processed"
EXPECTED_COLUMNS = [
    "model", "group", "n_train", "n_val",
    "within_5pct", "within_10pct", "mae_usd_m", "mae_log", "r2_log",
    "years_bucket_accuracy", "years_balanced_accuracy", "years_mae",
    "aav_learning_rate", "aav_max_depth", "aav_max_iter",
    "years_learning_rate", "years_max_depth", "years_max_iter"]


def _prepared(group):
    parts = bt.load_group(group)
    X_tr = bt.build_X(parts["train"], group)
    X_va = bt.align_categories(X_tr, bt.build_X(parts["val"], group))
    return parts, X_tr, X_va


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_no_contract_outcome_columns_used(group):
    assert not set(bt.FEATURES[group]) & bt.EXCLUDED
    X = bt.build_X(bt.load_group(group)["train"], group)
    assert list(X.columns) == bt.FEATURES[group]


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_no_2024_2025_rows_loaded(group):
    parts = bt.load_group(group)
    assert set(parts) == {"train", "val"}
    assert all(p["offseason"].max() <= 2023 for p in parts.values())


def test_share_target_converts_back_to_usd_m():
    df = pd.DataFrame({"aav_nominal": [0.74, 12.5, 43.33], "offseason": [2014, 2019, 2023]})

    class Exact:  # a "model" that predicts the true log share
        def predict(self, X):
            return bt.log_share(df)

    pred = bt.predict_aav(Exact(), df, df["offseason"], max_log_share=np.inf)
    assert pred == pytest.approx(df["aav_nominal"].to_numpy())
    assert bt.log_share(df) == pytest.approx(np.log(to_share(df["aav_nominal"], df["offseason"])))


def test_predictions_never_exceed_training_max_share():
    class TooHigh:
        def predict(self, X):
            return np.array([0.0, -1.0, -5.0])  # share 1.0 (a whole CBT threshold), 0.37, 0.007

    offseason = pd.Series([2020, 2022, 2023])
    cap = np.log(0.2)
    pred = bt.predict_aav(TooHigh(), None, offseason, cap)
    assert (to_share(pred, offseason) <= 0.2 + 1e-12).all()
    assert to_share(pred, offseason)[2] == pytest.approx(np.exp(-5.0))  # below the cap: unchanged


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_real_predictions_positive_and_capped(group):
    parts, X_tr, X_va = _prepared(group)
    y = bt.log_share(parts["train"])
    pred = bt.predict_aav(bt.fit_aav(X_tr, y, max_iter=20), X_va, parts["val"]["offseason"], y.max())
    assert pred.shape == (len(X_va),) and (pred > 0).all()
    assert (np.log(to_share(pred, parts["val"]["offseason"])) <= y.max() + 1e-12).all()
    classes = bt.fit_years(X_tr, parts["train"]["years_new"], max_iter=20).predict(X_va)
    assert set(classes) <= {0, 1, 2, 3, 4}


def test_score_years_uses_class_midpoints():
    m = bt.score_years([1, 2, 6], [0, 2, 4])  # predicted classes 1, 3, 6+ years -> midpoints 1, 3, 7
    assert m["years_mae"] == pytest.approx((0 + 1 + 1) / 3)
    assert m["years_bucket_accuracy"] == pytest.approx(2 / 3)
    assert m["years_balanced_accuracy"] == pytest.approx(2 / 3)


EXPECTED_BUCKET = {
    "hitters": {"C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
                "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH"},
    "pitchers": {"SP": "SP", "RP": "RP", "P": "RP"},
}


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_position_bucket_covers_every_position_in_the_data(group):
    df = pd.read_csv(DATA / f"contracts_features_{group}.csv")
    firsts = set(df["position"].astype(str).str.split("/").str[0].str.strip())
    assert firsts == set(EXPECTED_BUCKET[group])  # a new position value must be mapped on purpose
    for pos in df["position"].unique():
        first = str(pos).split("/")[0].strip()
        assert bt.position_bucket(pos, group) == EXPECTED_BUCKET[group][first]


def test_position_bucket_rejects_unknown_hitter_position():
    with pytest.raises(ValueError):
        bt.position_bucket("XX", "hitters")


def test_results_csv_in_d19_order(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    monkeypatch.setattr(bt, "GRID", {"learning_rate": [0.1], "max_depth": [3], "max_iter": [20]})
    (tmp_path / "data/processed").mkdir(parents=True)
    for g in bt.GROUPS:
        name = f"contracts_features_{g}.csv"
        (tmp_path / "data/processed" / name).write_bytes((DATA / name).read_bytes())
    bt.main()
    out = pd.read_csv(tmp_path / "results/boosted_tree_val_metrics.csv")
    assert list(out.columns) == EXPECTED_COLUMNS
    assert "years_balanced_accuracy" in out.columns and out["years_balanced_accuracy"].between(0, 1).all()
    assert list(out["group"]) == ["hitters", "pitchers"]
    assert (out["model"] == "boosted_tree").all()


def test_v3_grid_unchanged_and_v4_adds_001():
    assert bt.GRIDS[3] == {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4], "max_iter": [300, 600]}
    assert bt.GRIDS[4] == {**bt.GRIDS[3], "learning_rate": [0.01, 0.03, 0.05, 0.1]}
    assert bt.LATEST == 4 and bt.GRID == bt.GRIDS[4]


def test_each_version_writes_its_own_d19_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "ROOT", tmp_path)
    small = {"learning_rate": [0.1], "max_depth": [3], "max_iter": [20]}
    monkeypatch.setattr(bt, "GRIDS", {3: small, 4: small})
    monkeypatch.setattr(bt, "GRID", small)
    (tmp_path / "data/processed").mkdir(parents=True)
    for g in bt.GROUPS:
        name = f"contracts_features_{g}.csv"
        (tmp_path / "data/processed" / name).write_bytes((DATA / name).read_bytes())
    bt.main(3)
    bt.main(4)
    for name in ("boosted_tree_v3_val_metrics.csv", "boosted_tree_val_metrics.csv"):
        out = pd.read_csv(tmp_path / "results" / name)
        assert list(out.columns) == EXPECTED_COLUMNS
        assert list(out["group"]) == ["hitters", "pitchers"]


def test_model_history_has_one_v4_row_per_group():
    hist = pd.read_csv(pathlib.Path(__file__).parents[1] / "results/model_history.csv")
    v4 = hist[(hist["model"] == "boosted_tree") & (hist["version"] == 4)]
    assert sorted(v4["group"]) == ["hitters", "pitchers"]
    latest = pd.read_csv(pathlib.Path(__file__).parents[1] / "results/boosted_tree_val_metrics.csv")
    for _, row in latest.iterrows():
        h = v4[v4["group"] == row["group"]].iloc[0]
        assert h["r2_log"] == pytest.approx(row["r2_log"], abs=1e-4)
        assert h["years_balanced_accuracy"] == pytest.approx(row["years_balanced_accuracy"], abs=1e-4)
