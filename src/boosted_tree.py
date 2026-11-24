"""Boosted-tree baseline for contract length and nominal AAV (issue #23, D-09).

Four HistGradientBoostingRegressor models: hitters and pitchers, each predicting `years_new`
and `aav_nominal`. AAV is fit on log(AAV) and converted back to $M for scoring. Settings come
from a small fixed grid, chosen on validation (2022-2023); every model is fit on train
(2014-2021) only. The test classes (2024-2025) are never loaded here.

The feature list is FEATURES below. It is meant to be easy to change so every model can use the
same inputs; columns in EXCLUDED must never be features (they describe the new contract itself).

Usage: python src/boosted_tree.py   -> results/boosted_tree_val_metrics.csv (format: D-18)
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from metrics import report
from splits import split_by_group

ROOT = Path(__file__).parents[1]
GROUPS = {"hitters": "hitter", "pitchers": "pitcher"}
TARGETS = ("years_new", "aav_nominal")

# Shared input list (D-18). service_years_at_signing is not used: it is blank in every row.
COMMON = ["age_at_signing", "contract_type", "position_bucket", "seasons_in_window",
          "won_mvp", "won_cy_young", "won_gold_glove", "won_silver_slugger",
          "has_prev_contract", "prev_contract_type", "prev_contract_years",
          "prev_contract_aav", "years_since_prev_contract"]
STATS = {"hitters": ["war_3yr", "r_3yr", "ba_3yr", "ops_plus_3yr"],
         "pitchers": ["war_3yr", "era_plus_3yr", "fip_3yr", "whip_3yr"]}
FEATURES = {g: COMMON + STATS[g] for g in GROUPS}
CATEGORICAL = ["contract_type", "position_bucket", "prev_contract_type"]
# Position bucket from the first listed position (D-18). Pitchers: SP, everything else RP.
HITTER_BUCKET = {"C": "C", "1B": "IF", "2B": "IF", "3B": "IF", "SS": "IF",
                 "LF": "OF", "CF": "OF", "RF": "OF", "OF": "OF", "DH": "DH"}

# Terms of the new contract, the signing team, and identifiers: never features.
EXCLUDED = {"years_new", "total_value_nominal", "aav_nominal", "aav_adjusted", "deferred_amount_m",
            "aav_adjusted_method", "at_league_minimum", "has_opt_out", "has_deferral",
            "has_incentives", "team", "is_resign", "player_name", "mlb_id", "fg_id", "bbref_id",
            "source_url", "notes", "signing_date", "offseason", "prev_team"}

GRID = {"learning_rate": [0.05, 0.1], "max_depth": [3, 5], "max_iter": [100, 300]}
# Same length classes as the neural network (PR #24): 1, 2, 3, 4-5, 6+ years.
YEAR_BUCKET_EDGES = [1.5, 2.5, 3.5, 5.5]
METRIC_COLUMNS = ["within_5pct", "within_10pct", "mae_usd_m", "mae_log", "r2_log",
                  "years_bucket_accuracy", "years_mae"]
SETTING_COLUMNS = [f"{t}_{k}" for t in ("aav", "years") for k in GRID]
RESULT_COLUMNS = ["model", "group", "n_train", "n_val", *METRIC_COLUMNS, "exact_year", *SETTING_COLUMNS]
SHORT = {"aav_nominal": "aav", "years_new": "years"}


def load_group(group: str) -> dict[str, pd.DataFrame]:
    """Train and validation rows for one group. Test rows are dropped before anything is returned."""
    df = pd.read_csv(ROOT / "data/processed" / f"contracts_features_{group}.csv")
    parts = split_by_group(df, GROUPS[group])
    return {"train": parts["train"], "val": parts["val"]}


def position_bucket(position: str, group: str) -> str:
    """First listed position, grouped: hitters C / IF / OF / DH, pitchers SP / RP (D-18)."""
    first = str(position).split("/")[0].strip()
    if group == "pitchers":
        return "SP" if first == "SP" else "RP"
    if first not in HITTER_BUCKET:
        raise ValueError(f"unknown hitter position {position!r}")
    return HITTER_BUCKET[first]


def build_X(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Feature frame in a fixed column order; categoricals as pandas categories, NaN left as is."""
    df = df.assign(position_bucket=[position_bucket(p, group) for p in df["position"]])
    X = df[FEATURES[group]].copy()
    for col in X.columns:
        if col in CATEGORICAL:
            X[col] = X[col].astype("category")
        else:
            X[col] = X[col].astype(float)
    return X


def align_categories(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Give X_other the train categories (an unseen level becomes NaN)."""
    out = X_other.copy()
    for col in CATEGORICAL:
        out[col] = pd.Categorical(out[col], categories=X_train[col].cat.categories)
    return out


def fit(X: pd.DataFrame, y: np.ndarray, **params) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0, **params)
    return model.fit(X, y)


def score_years(y_true, y_pred) -> dict[str, float]:
    """MAE in years, exact-year share (rounded), and length-class accuracy."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rounded = np.clip(np.rint(y_pred), 1, None)
    return {"years_mae": float(np.abs(y_pred - y_true).mean()),
            "exact_year": float((rounded == y_true).mean()),
            "years_bucket_accuracy": float((np.digitize(rounded, YEAR_BUCKET_EDGES)
                                 == np.digitize(y_true, YEAR_BUCKET_EDGES)).mean())}


def predict(model, X: pd.DataFrame, target: str) -> np.ndarray:
    pred = model.predict(X)
    return np.exp(pred) if target == "aav_nominal" else np.clip(pred, 1, None)


def score(target: str, y_true, y_pred) -> dict[str, float]:
    return report(y_true, y_pred) if target == "aav_nominal" else score_years(y_true, y_pred)


def tune(group: str) -> dict:
    """Pick grid settings on validation for both targets of one group; one D-18 result row."""
    parts = load_group(group)
    X_tr = build_X(parts["train"], group)
    X_va = align_categories(X_tr, build_X(parts["val"], group))
    row = {"model": "boosted_tree", "group": group, "n_train": len(X_tr), "n_val": len(X_va)}
    for target in TARGETS:
        y_tr, y_va = parts["train"][target].to_numpy(float), parts["val"][target].to_numpy(float)
        fit_y = np.log(y_tr) if target == "aav_nominal" else y_tr
        key = "mae_log" if target == "aav_nominal" else "years_mae"
        best = None
        for values in product(*GRID.values()):
            params = dict(zip(GRID, values))
            metrics = score(target, y_va, predict(fit(X_tr, fit_y, **params), X_va, target))
            if best is None or metrics[key] < best[1][key]:
                best = (params, metrics)
        row.update(best[1])
        row.update({f"{SHORT[target]}_{k}": v for k, v in best[0].items()})
    return row


def main() -> None:
    results = pd.DataFrame([tune(g) for g in GROUPS])[RESULT_COLUMNS]
    out = ROOT / "results/boosted_tree_val_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(results.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
