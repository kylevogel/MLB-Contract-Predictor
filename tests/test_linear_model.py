import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
import linear_model as lm

POSITIONS = {"hitter": ["C", "1B/DH", "SS", "CF/RF"], "pitcher": ["SP/SP2", "RP/CL", "SP", "P"]}


def fake_group(group: str, seed: int) -> pd.DataFrame:
    """Made-up contracts for one group: 4 per offseason 2014-2025, AAV rising with WAR."""
    rng = np.random.default_rng(seed)
    n = 4 * 12
    df = pd.DataFrame({
        "offseason": np.repeat(np.arange(2014, 2026), 4),
        "pos_group": group,
        "position": POSITIONS[group] * 12,
        "war_3yr": rng.uniform(0, 6, n),
        "age_at_signing": rng.uniform(24, 36, n),
        "seasons_in_window": 3, "won_mvp": 0, "won_cy_young": 0, "won_gold_glove": 0, "won_silver_slugger": 0,
        "has_prev_contract": False,
        "prev_contract_years": np.nan, "prev_contract_aav": np.nan, "years_since_prev_contract": np.nan,
        "contract_type": rng.choice(["free_agent", "extension_arb"], n), "prev_contract_type": np.nan,
    })
    for col in lm.STATS[group][1:]:
        df[col] = rng.uniform(50, 150, n)
    df.loc[0, lm.STATS[group][1]] = np.nan  # a blank stat must be imputed, not crash
    df["aav_nominal"] = np.exp(0.5 + 0.4 * df.war_3yr + rng.normal(0, 0.1, n))
    df["years_new"] = (1 + df.war_3yr // 2).astype(int)
    return df


@pytest.fixture(scope="module")
def data():
    return {"hitter": fake_group("hitter", 1), "pitcher": fake_group("pitcher", 2)}


@pytest.fixture(scope="module")
def fitted(data):
    return lm.fit_linear_models(data)


def test_four_models_with_alpha_from_grid(fitted):
    assert set(fitted) == {(g, t) for g in lm.GROUPS for t in lm.TARGETS}
    assert all(alpha in lm.ALPHAS for _, alpha in fitted.values())


def test_predictions_on_original_scale(fitted, data):
    for group in lm.GROUPS:
        train, val = lm.train_val(data[group], group)
        X = pd.concat([train, val])[lm.feature_columns(group)]
        aav = fitted[(group, "aav")][0].predict(X)
        years = fitted[(group, "years")][0].predict(X)
        assert (aav > 0).all() and (years >= 1).all()
        # learns the made-up signal: more WAR -> higher AAV
        assert np.corrcoef(pd.concat([train, val]).war_3yr, aav)[0, 1] > 0.9


def test_test_classes_never_used(data, monkeypatch):
    seen = []
    real_fit = lm.fit_one
    monkeypatch.setattr(lm, "fit_one", lambda tr, va, g, t: seen.append((tr, va)) or real_fit(tr, va, g, t))
    lm.fit_linear_models(data)
    lm.val_metrics(data)
    assert seen and all(tr.offseason.max() <= 2021 and va.offseason.between(2022, 2023).all() for tr, va in seen)


def test_metrics_ignore_test_rows(data):
    without_test = {g: df[df.offseason <= 2023] for g, df in data.items()}
    poisoned = {g: df.assign(aav_nominal=np.where(df.offseason >= 2024, 999.0, df.aav_nominal))
                for g, df in data.items()}
    pd.testing.assert_frame_equal(lm.val_metrics(without_test), lm.val_metrics(poisoned))


def test_no_leakage_columns():
    leaky = {"is_resign", "aav_nominal", "years_new", "total_value_nominal", "aav_adjusted", "deferred_amount_m",
             "at_league_minimum", "has_deferral", "has_opt_out", "has_incentives", "offseason", "team"}
    for group in lm.GROUPS:
        assert not leaky & set(lm.feature_columns(group))
        assert "position_bucket" in lm.feature_columns(group)


def test_each_group_trained_only_on_its_own_rows(data, monkeypatch):
    seen = []
    real_fit = lm.fit_one
    monkeypatch.setattr(lm, "fit_one", lambda tr, va, g, t: seen.append((g, tr, va)) or real_fit(tr, va, g, t))
    lm.fit_linear_models(data)
    assert {g for g, _, _ in seen} == set(lm.GROUPS)
    for group, tr, va in seen:
        assert set(tr.pos_group) == {group} and set(va.pos_group) == {group}


def test_position_bucket_mapping():
    assert [lm.position_bucket(p, "hitter") for p in ["C", "1B/DH", "2B", "3B", "SS", "LF", "CF", "RF/LF", "OF", "DH"]] \
        == ["C", "IF", "IF", "IF", "IF", "OF", "OF", "OF", "OF", "DH"]
    assert [lm.position_bucket(p, "pitcher") for p in ["SP/SP2", "RP/CL", "P"]] == ["SP", "RP", "RP"]
    with pytest.raises(ValueError):
        lm.position_bucket("XX", "hitter")


@pytest.mark.parametrize("group", lm.GROUPS)
def test_position_bucket_covers_real_data(group):
    df = pd.read_csv(lm.ROOT / f"data/processed/contracts_features_{group}s.csv")
    buckets = {lm.position_bucket(p, group) for p in df["position"].unique()}  # raises on an unknown value
    assert buckets == ({"C", "IF", "OF", "DH"} if group == "hitter" else {"SP", "RP"})


def test_years_bucket_edges():
    assert lm.years_bucket([1, 2, 3, 4, 5, 6, 15]).tolist() == [0, 1, 2, 3, 3, 4, 4]


def test_val_metrics_columns(data):
    out = lm.val_metrics(data)
    assert list(out.columns) == lm.METRIC_COLUMNS
    assert list(out.group) == ["hitters", "pitchers"] and set(out.model) == {"linear"}
    assert (out.n_val == 8).all() and (out.n_train == 32).all()  # 2 val classes, 8 train classes, 4 rows each


def test_committed_results_csv_columns():
    out = pd.read_csv(lm.RESULTS)
    assert list(out.columns) == lm.METRIC_COLUMNS
    assert list(out.group) == ["hitters", "pitchers"]


# --- D-19 linear model (#38) ---
from salary_scale import from_share, to_share


@pytest.fixture(scope="module")
def data_d19(data):
    """The made-up data above, plus one hitter with a previous contract."""
    out = {g: df.copy() for g, df in data.items()}
    out["hitter"].loc[1, ["has_prev_contract", "prev_contract_years", "prev_contract_aav",
                          "years_since_prev_contract"]] = [True, 3, 12.0, 3]
    return out


@pytest.fixture(scope="module")
def fitted_d19(data_d19):
    return lm.fit_linear_models_d19(data_d19)


def test_d19_params_from_grids(fitted_d19):
    assert set(fitted_d19) == set(lm.GROUPS)
    for fit in fitted_d19.values():
        assert fit["alpha_aav"] in lm.ALPHAS_D19 and fit["C_years"] in lm.CS_D19


def test_d19_share_target_converts_back_to_usd_m(fitted_d19, data_d19):
    for group in lm.GROUPS:
        train, val = lm.train_val_d19(data_d19[group], group)
        assert np.allclose(from_share(np.exp(lm.log_share(train)), train.offseason), train.aav_nominal)
        fit = fitted_d19[group]
        rows = pd.concat([train, val])
        aav = lm.predict_aav_d19(fit["aav_model"], rows, fit["cap"])
        log_pred = fit["aav_model"].predict(rows[lm.feature_columns_d19(group)])
        assert np.allclose(aav, from_share(np.exp(np.minimum(log_pred, fit["cap"])), rows.offseason))
        assert np.corrcoef(np.log(rows.aav_nominal), np.log(aav))[0, 1] > 0.9  # learns the made-up signal in $M


def test_d19_predictions_never_exceed_training_max_share(fitted_d19, data_d19):
    for group in lm.GROUPS:
        train, val = lm.train_val_d19(data_d19[group], group)
        fit = fitted_d19[group]
        assert fit["cap"] == pytest.approx(lm.log_share(train).max())
        extreme = val.assign(war_3yr=1000.0)  # far outside training: uncapped, the prediction would explode
        share = to_share(lm.predict_aav_d19(fit["aav_model"], extreme, fit["cap"]), extreme.offseason)
        assert (share <= np.exp(fit["cap"]) + 1e-12).all()


def test_d19_length_is_a_class_prediction(fitted_d19, data_d19):
    for group in lm.GROUPS:
        _, val = lm.train_val_d19(data_d19[group], group)
        assert set(fitted_d19[group]["years_model"].predict(val[lm.feature_columns_d19(group)])) <= set(range(5))


def test_d19_log1p_prev_contract_aav(data_d19):
    train, _ = lm.train_val_d19(data_d19["hitter"], "hitter")
    assert train.loc[1, "log1p_prev_contract_aav"] == pytest.approx(np.log1p(12.0))
    assert (train.loc[~train.has_prev_contract.astype(bool), "log1p_prev_contract_aav"] == 0).all()
    for group in lm.GROUPS:
        cols = lm.feature_columns_d19(group)
        assert "log1p_prev_contract_aav" in cols and "prev_contract_aav" not in cols and "is_resign" not in cols


def test_d19_test_classes_never_used(data_d19, monkeypatch):
    seen = []
    real_fit = lm.fit_group_d19
    monkeypatch.setattr(lm, "fit_group_d19", lambda tr, va, g: seen.append((g, tr, va)) or real_fit(tr, va, g))
    lm.fit_linear_models_d19(data_d19)
    lm.val_metrics_d19(data_d19)
    assert {g for g, _, _ in seen} == set(lm.GROUPS)
    for group, tr, va in seen:
        assert tr.offseason.max() <= 2021 and va.offseason.between(2022, 2023).all()
        assert set(tr.pos_group) == {group} and set(va.pos_group) == {group}


def test_d19_metrics_ignore_test_rows(data_d19):
    without_test = {g: df[df.offseason <= 2023] for g, df in data_d19.items()}
    poisoned = {g: df.assign(aav_nominal=np.where(df.offseason >= 2024, 999.0, df.aav_nominal),
                             years_new=np.where(df.offseason >= 2024, 15, df.years_new))
                for g, df in data_d19.items()}
    pd.testing.assert_frame_equal(lm.val_metrics_d19(without_test), lm.val_metrics_d19(poisoned))


def test_d19_val_metrics_columns(data_d19):
    out = lm.val_metrics_d19(data_d19)
    assert list(out.columns) == lm.METRIC_COLUMNS_D19
    assert lm.METRIC_COLUMNS_D19[9:12] == ["years_bucket_accuracy", "years_balanced_accuracy", "years_mae"]
    assert list(out.group) == ["hitters", "pitchers"] and set(out.model) == {"linear"}
    assert out.years_balanced_accuracy.between(0, 1).all()


def test_d19_committed_results_csv_columns():
    out = pd.read_csv(lm.RESULTS_D19)
    assert list(out.columns) == lm.METRIC_COLUMNS_D19 and "years_balanced_accuracy" in out.columns
    assert list(out.group) == ["hitters", "pitchers"]
