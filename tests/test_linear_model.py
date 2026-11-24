import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest
import linear_model as lm


def fake_group(group: str, seed: int) -> pd.DataFrame:
    """Made-up contracts for one group: 4 per offseason 2014-2025, AAV rising with WAR."""
    rng = np.random.default_rng(seed)
    n = 4 * 12
    df = pd.DataFrame({
        "offseason": np.repeat(np.arange(2014, 2026), 4),
        "pos_group": group,
        "war_3yr": rng.uniform(0, 6, n),
        "age_at_signing": rng.uniform(24, 36, n),
        "seasons_in_window": 3, "won_mvp": 0, "won_cy_young": 0, "won_gold_glove": 0, "won_silver_slugger": 0,
        "is_resign": rng.integers(0, 2, n).astype(bool), "has_prev_contract": False,
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
        X = data[group][lm.feature_columns(group)]
        aav = fitted[(group, "aav")][0].predict(X)
        years = fitted[(group, "years")][0].predict(X)
        assert (aav > 0).all() and (years >= 1).all()
        # learns the made-up signal: more WAR -> higher AAV
        assert np.corrcoef(data[group].war_3yr, aav)[0, 1] > 0.9


def test_test_classes_never_used(data, monkeypatch):
    seen = []
    real_fit = lm.fit_one
    monkeypatch.setattr(lm, "fit_one", lambda tr, va, g, t: seen.append((tr, va)) or real_fit(tr, va, g, t))
    lm.fit_linear_models(data)
    assert seen and all(tr.offseason.max() <= 2021 and va.offseason.between(2022, 2023).all() for tr, va in seen)


def test_no_leakage_columns():
    leaky = {"aav_nominal", "years_new", "total_value_nominal", "aav_adjusted", "deferred_amount_m",
             "at_league_minimum", "has_deferral", "has_opt_out", "has_incentives"}
    for group in lm.GROUPS:
        assert not leaky & set(lm.feature_columns(group))
