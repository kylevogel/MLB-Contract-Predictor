"""Linear baseline (issue #18): ridge regression for AAV and contract length, hitters and pitchers separately.

Four models: {hitter, pitcher} x {aav, years}. AAV is fit on log(aav_nominal) and predicted back in $M;
years_new is fit directly and predictions are held at >= 1 year. The ridge penalty `alpha` is chosen
from ALPHAS by the lowest validation error (2022-2023) after fitting on train (2014-2021).
Test classes (2024-2025) are dropped before anything is fit or evaluated.

Inputs and results format follow DECISIONS.md D-18 (shared by all three models). Validation metrics
are written to results/linear_val_metrics.csv, one row per group.

Usage: python src/linear_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from metrics import report
from splits import split_by_offseason

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results/linear_val_metrics.csv"
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GROUPS = ("hitter", "pitcher")
TARGETS = {"aav": "aav_nominal", "years": "years_new"}

STATS = {"hitter": ["war_3yr", "r_3yr", "ba_3yr", "ops_plus_3yr"],
         "pitcher": ["war_3yr", "era_plus_3yr", "fip_3yr", "whip_3yr"]}
SHARED = ["age_at_signing", "seasons_in_window", "won_mvp", "won_cy_young", "won_gold_glove",
          "won_silver_slugger", "has_prev_contract"]
PREV = ["prev_contract_years", "prev_contract_aav", "years_since_prev_contract"]  # blank = no previous contract
CATEGORICAL = ["contract_type", "prev_contract_type", "position_bucket"]

HITTER_BUCKET = {"C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
                 "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH"}
METRIC_COLUMNS = ["model", "group", "n_train", "n_val", "within_5pct", "within_10pct", "mae_usd_m",
                  "mae_log", "r2_log", "years_bucket_accuracy", "years_mae", "alpha_aav", "alpha_years"]


def feature_columns(group: str) -> list[str]:
    """Every input column for one group (D-18). Nothing computed from the contract being predicted."""
    return STATS[group] + SHARED + PREV + CATEGORICAL


def position_bucket(position: str, group: str) -> str:
    """First listed position, grouped (D-18): hitters C/IF/OF/DH, pitchers SP/RP (anything not SP -> RP)."""
    first = str(position).split("/")[0].strip()
    if group == "pitcher":
        return "SP" if first == "SP" else "RP"
    if first not in HITTER_BUCKET:
        raise ValueError(f"unknown hitter position {position!r}")
    return HITTER_BUCKET[first]


def years_bucket(years) -> np.ndarray:
    """Contract length class: 1 -> 0, 2 -> 1, 3 -> 2, 4-5 -> 3, 6+ -> 4."""
    y = np.asarray(years, dtype=int)
    return np.select([y <= 1, y == 2, y == 3, y <= 5], [0, 1, 2, 3], default=4)


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


def load_data() -> dict[str, pd.DataFrame]:
    return {g: pd.read_csv(ROOT / f"data/processed/contracts_features_{g}s.csv") for g in GROUPS}


def train_val(df: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the position bucket and split; the test classes are dropped here and never returned."""
    df = df.assign(position_bucket=df["position"].map(lambda p: position_bucket(p, group)))
    parts = split_by_offseason(df)
    return parts["train"], parts["val"]


def fit_linear_models(data: dict[str, pd.DataFrame] | None = None) -> dict[tuple[str, str], tuple]:
    """Fit all four models. `data` maps group -> feature dataset (default: the files in data/processed).
    Returns {(group, target): (fitted model, chosen alpha)}."""
    data = load_data() if data is None else data
    fitted = {}
    for group in GROUPS:
        train, val = train_val(data[group], group)
        for target in TARGETS:
            fitted[(group, target)] = fit_one(train, val, group, target)
    return fitted


def val_metrics(data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fit per group and score validation (D-18 format): AAV metrics in $M, then contract-length metrics."""
    data = load_data() if data is None else data
    rows = []
    for group in GROUPS:
        train, val = train_val(data[group], group)
        (aav_model, alpha_aav), (years_model, alpha_years) = (fit_one(train, val, group, t) for t in TARGETS)
        X = val[feature_columns(group)]
        years_pred = years_model.predict(X)
        rounded = np.maximum(np.floor(years_pred + 0.5), 1)
        rows.append({
            "model": "linear", "group": f"{group}s", "n_train": len(train), "n_val": len(val),
            **report(val["aav_nominal"], aav_model.predict(X)),
            "years_bucket_accuracy": float((years_bucket(rounded) == years_bucket(val["years_new"])).mean()),
            "years_mae": float(np.abs(years_pred - val["years_new"]).mean()),
            "alpha_aav": alpha_aav, "alpha_years": alpha_years,
        })
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def main() -> None:
    results = val_metrics()
    RESULTS.parent.mkdir(exist_ok=True)
    results.to_csv(RESULTS, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
