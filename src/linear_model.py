"""Linear baseline (issue #18): ridge regression for AAV and contract length, hitters and pitchers separately.

Four models: {hitter, pitcher} x {aav, years}. AAV is fit on log(aav_nominal) and predicted back in $M;
years_new is fit directly and predictions are held at >= 1 year. The ridge penalty `alpha` is chosen
from ALPHAS by the lowest validation error (2022-2023) after fitting on train (2014-2021).
Test classes (2024-2025) are dropped before anything is fit or evaluated.

Inputs and results format follow DECISIONS.md D-18 (shared by all three models). Validation metrics
are written to results/linear_val_metrics.csv, one row per group.

Two versions live in this file: the original D-18 model above, kept unchanged for comparison, and the
D-19 model (#38) below it. `main()` writes the D-19 results to results/linear_val_metrics.csv and the
original model's to results/linear_d18_val_metrics.csv.

Usage: python src/linear_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import metrics
from metrics import report
from salary_scale import from_share, to_share
from splits import split_by_offseason

# --- D-18 linear model (original, kept for comparison) ---

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results/linear_d18_val_metrics.csv"
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


# --- D-19 linear model (#38) ---
# AAV: ridge on log(AAV share of the class's CBT threshold), capped at the group's training-max log share,
# converted back to $M (D-19). Length: class-balanced logistic regression on the classes 1/2/3/4-5/6+.
# The previous contract's AAV enters as log1p (0 when there is none); other inputs as D-18.

RESULTS_D19 = ROOT / "results/linear_val_metrics.csv"
ALPHAS_D19 = [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
CS_D19 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
YEAR_MIDPOINTS = np.array([1.0, 2.0, 3.0, 4.5, 7.0])  # class 1 / 2 / 3 / 4-5 / 6+ in years
PREV_D19 = ["prev_contract_years", "log1p_prev_contract_aav", "years_since_prev_contract"]
METRIC_COLUMNS_D19 = ["model", "group", "n_train", "n_val", "within_5pct", "within_10pct", "mae_usd_m", "mae_log",
                      "r2_log", "years_bucket_accuracy", "years_balanced_accuracy", "years_mae", "alpha_aav", "C_years"]


def feature_columns_d19(group: str) -> list[str]:
    """D-18 inputs with the raw previous AAV swapped for its log1p."""
    return STATS[group] + SHARED + PREV_D19 + CATEGORICAL


def _preprocess_d19(group: str) -> ColumnTransformer:
    """Same imputation and scaling as the D-18 model (learned on train rows only), with the D-19 inputs."""
    return ColumnTransformer([
        ("stats", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()),
         STATS[group] + SHARED),
        ("prev", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, keep_empty_features=True),
                               StandardScaler()), PREV_D19),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="none", keep_empty_features=True),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL),
    ])


def train_val_d19(df: pd.DataFrame, group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The D-18 split (test classes dropped) plus the log1p previous-AAV input."""
    df = df.assign(log1p_prev_contract_aav=np.log1p(df["prev_contract_aav"].fillna(0)))
    return train_val(df, group)


def log_share(df: pd.DataFrame) -> np.ndarray:
    """The D-19 AAV target: log of AAV as a share of the class's CBT threshold."""
    return np.log(to_share(df["aav_nominal"], df["offseason"]))


def predict_aav_d19(model: Pipeline, df: pd.DataFrame, cap: float) -> np.ndarray:
    """Predicted AAV in $M: log share capped at `cap` (training max), then back to $M for the row's class."""
    pred = np.minimum(model.predict(df[model.feature_names_in_]), cap)
    return from_share(np.exp(pred), df["offseason"])


def fit_aav_d19(train: pd.DataFrame, val: pd.DataFrame, group: str) -> tuple[Pipeline, float, float]:
    """Try every alpha on train, keep the lowest validation log-AAV error. Returns (model, alpha, cap)."""
    cols, y = feature_columns_d19(group), log_share(train)
    cap, best = float(y.max()), None
    for alpha in ALPHAS_D19:
        model = Pipeline([("prep", _preprocess_d19(group)), ("ridge", Ridge(alpha=alpha))]).fit(train[cols], y)
        err = metrics.mae_log(val["aav_nominal"], predict_aav_d19(model, val, cap))
        if best is None or err < best[0]:
            best = (err, alpha, model)
    return best[2], best[1], cap


def fit_years_d19(train: pd.DataFrame, val: pd.DataFrame, group: str) -> tuple[Pipeline, float]:
    """Try every C on train, keep the highest validation years_balanced_accuracy. Returns (model, C)."""
    cols, best = feature_columns_d19(group), None
    for C in CS_D19:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
        model = Pipeline([("prep", _preprocess_d19(group)), ("logit", clf)])
        model.fit(train[cols], metrics.years_bucket(train["years_new"]))
        score = metrics.years_class_report(val["years_new"], model.predict(val[cols]))["years_balanced_accuracy"]
        if best is None or score > best[0]:
            best = (score, C, model)
    return best[2], best[1]


def fit_group_d19(train: pd.DataFrame, val: pd.DataFrame, group: str) -> dict:
    aav_model, alpha, cap = fit_aav_d19(train, val, group)
    years_model, C = fit_years_d19(train, val, group)
    return {"aav_model": aav_model, "alpha_aav": alpha, "cap": cap, "years_model": years_model, "C_years": C}


def fit_linear_models_d19(data: dict[str, pd.DataFrame] | None = None) -> dict[str, dict]:
    """Fit the D-19 AAV and length models for each group. Returns {group: fit_group_d19(...) result}."""
    data = load_data() if data is None else data
    return {g: fit_group_d19(*train_val_d19(data[g], g), g) for g in GROUPS}


def val_metrics_d19(data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fit per group and score validation in the D-19 format: AAV in $M, then contract-length classes."""
    data = load_data() if data is None else data
    rows = []
    for group in GROUPS:
        train, val = train_val_d19(data[group], group)
        fit = fit_group_d19(train, val, group)
        pred_class = fit["years_model"].predict(val[feature_columns_d19(group)])
        rows.append({
            "model": "linear", "group": f"{group}s", "n_train": len(train), "n_val": len(val),
            **report(val["aav_nominal"], predict_aav_d19(fit["aav_model"], val, fit["cap"])),
            **metrics.years_class_report(val["years_new"], pred_class),
            "years_mae": float(np.abs(YEAR_MIDPOINTS[pred_class] - val["years_new"]).mean()),
            "alpha_aav": fit["alpha_aav"], "C_years": fit["C_years"],
        })
    return pd.DataFrame(rows, columns=METRIC_COLUMNS_D19)


def main() -> None:
    RESULTS.parent.mkdir(exist_ok=True)
    for name, path, table in (("D-18 (original)", RESULTS, val_metrics()), ("D-19", RESULTS_D19, val_metrics_d19())):
        table.to_csv(path, index=False)
        print(f"{name} -> {path.relative_to(ROOT)}\n{table.to_string(index=False)}\n")


if __name__ == "__main__":
    main()
