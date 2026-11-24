"""Linear baseline (issue #18): ridge regression for AAV and contract length, hitters and pitchers separately.

Four models: {hitter, pitcher} x {aav, years}. AAV is fit on log(aav_nominal) and predicted back in $M;
years_new is fit directly and predictions are held at >= 1 year. The ridge penalty `alpha` is chosen
from ALPHAS by the lowest validation error (2022-2023) after fitting on train (2014-2021).
Test classes (2024-2025) are dropped before anything is fit. Evaluation is out of scope here.

Usage: python src/linear_model.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from splits import split_by_offseason

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GROUPS = ("hitter", "pitcher")
TARGETS = {"aav": "aav_nominal", "years": "years_new"}

STATS = {"hitter": ["war_3yr", "r_3yr", "ba_3yr", "ops_plus_3yr"],
         "pitcher": ["war_3yr", "era_plus_3yr", "fip_3yr", "whip_3yr"]}
SHARED = ["seasons_in_window", "won_mvp", "won_cy_young", "won_gold_glove", "won_silver_slugger",
          "age_at_signing", "is_resign", "has_prev_contract"]
PREV = ["prev_contract_years", "prev_contract_aav", "years_since_prev_contract"]  # blank = no previous contract
CATEGORICAL = ["contract_type", "prev_contract_type"]


def feature_columns(group: str) -> list[str]:
    """Every input column for one group. Nothing computed from the contract being predicted."""
    return STATS[group] + SHARED + PREV + CATEGORICAL


def _identity(y):
    return y


def _at_least_one_year(y):
    return np.maximum(y, 1.0)


def make_model(group: str, target: str, alpha: float) -> TransformedTargetRegressor:
    """Impute and scale (learned on training rows only), then ridge; AAV on the log scale."""
    prep = ColumnTransformer([
        ("stats", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
         STATS[group] + SHARED),
        ("prev", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
                               StandardScaler()), PREV),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="none", keep_empty_features=True),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])
    pipe = Pipeline([("prep", prep), ("ridge", Ridge(alpha=alpha))])
    if target == "aav":
        return TransformedTargetRegressor(pipe, func=np.log, inverse_func=np.exp)
    return TransformedTargetRegressor(pipe, func=_identity, inverse_func=_at_least_one_year, check_inverse=False)


def _val_error(target: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    if target == "aav":
        return mean_absolute_error(np.log(y_true), np.log(y_pred))
    return mean_absolute_error(y_true, y_pred)


def fit_one(train: pd.DataFrame, val: pd.DataFrame, group: str, target: str):
    """Try every alpha on train, keep the one with the lowest validation error. Returns (model, alpha)."""
    cols, y_col = feature_columns(group), TARGETS[target]
    best = None
    for alpha in ALPHAS:
        model = make_model(group, target, alpha).fit(train[cols], train[y_col])
        err = _val_error(target, val[y_col], model.predict(val[cols]))
        if best is None or err < best[0]:
            best = (err, alpha, model)
    return best[2], best[1]


def fit_linear_models(data: dict[str, pd.DataFrame] | None = None) -> dict[tuple[str, str], tuple]:
    """Fit all four models. `data` maps group -> feature dataset (default: the files in data/processed).
    Returns {(group, target): (fitted model, chosen alpha)}."""
    if data is None:
        data = {g: pd.read_csv(f"data/processed/contracts_features_{g}s.csv") for g in GROUPS}
    fitted = {}
    for group in GROUPS:
        parts = split_by_offseason(data[group])  # the test split is never passed on
        for target in TARGETS:
            fitted[(group, target)] = fit_one(parts["train"], parts["val"], group, target)
    return fitted


def main() -> None:
    for (group, target), (_, alpha) in fit_linear_models().items():
        print(f"{group:8s} {target:5s} alpha={alpha}")


if __name__ == "__main__":
    main()
