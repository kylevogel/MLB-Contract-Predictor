"""Boosted-tree baseline for nominal AAV and contract length (issues #23, #36; D-18, D-19).

Hitters and pitchers are modelled separately. For each group:
- AAV: HistGradientBoostingRegressor on log(AAV share of the CBT threshold) (D-19a). Predicted log
  shares are capped at the largest training log share for the group (D-19b), converted back to $M
  with salary_scale.from_share, and scored on nominal $M with metrics.report().
- Length: HistGradientBoostingClassifier (class_weight="balanced") on metrics.years_bucket classes
  1 / 2 / 3 / 4-5 / 6+ (D-19c), scored with metrics.years_class_report and a midpoint years_mae.

Settings come from a fixed grid, chosen on validation (2022-2023): AAV by lowest mae_log, length by
highest years_balanced_accuracy. Every model is fit on train (2014-2021) only. The test classes
(2024-2025) are never loaded here.

Usage: python src/boosted_tree.py   -> results/boosted_tree_val_metrics.csv (format: D-19)
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from metrics import report, years_bucket, years_class_report
from salary_scale import from_share, to_share
from splits import split_by_group

ROOT = Path(__file__).parents[1]
GROUPS = {"hitters": "hitter", "pitchers": "pitcher"}

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

GRID = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4], "max_iter": [300, 600]}
SEED = 0
# Class midpoints in years for 1 / 2 / 3 / 4-5 / 6+, as the network uses (D-18).
YEAR_BUCKET_MIDPOINTS = np.array([1.0, 2.0, 3.0, 4.5, 7.0])
METRIC_COLUMNS = ["within_5pct", "within_10pct", "mae_usd_m", "mae_log", "r2_log",
                  "years_bucket_accuracy", "years_balanced_accuracy", "years_mae"]
SETTING_COLUMNS = [f"{t}_{k}" for t in ("aav", "years") for k in GRID]
RESULT_COLUMNS = ["model", "group", "n_train", "n_val", *METRIC_COLUMNS, *SETTING_COLUMNS]


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
        X[col] = X[col].astype("category") if col in CATEGORICAL else X[col].astype(float)
    return X


def align_categories(X_train: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Give X_other the train categories (an unseen level becomes NaN)."""
    out = X_other.copy()
    for col in CATEGORICAL:
        out[col] = pd.Categorical(out[col], categories=X_train[col].cat.categories)
    return out


def log_share(df: pd.DataFrame) -> np.ndarray:
    """AAV target: log of nominal AAV as a share of the class's CBT threshold (D-19a)."""
    return np.log(to_share(df["aav_nominal"], df["offseason"]))


def fit_aav(X: pd.DataFrame, y_log_share: np.ndarray, **params) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED, **params)
    return model.fit(X, y_log_share)


def predict_aav(model, X: pd.DataFrame, offseason, max_log_share: float) -> np.ndarray:
    """Predicted AAV in $M: log share capped at the training maximum (D-19b), then back to $M."""
    capped = np.minimum(model.predict(X), max_log_share)
    return from_share(np.exp(capped), offseason)


def fit_years(X: pd.DataFrame, years, **params) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(categorical_features="from_dtype", class_weight="balanced",
                                           random_state=SEED, **params)
    return model.fit(X, years_bucket(years))


def score_years(years_true, bucket_pred) -> dict[str, float]:
    """Class accuracy, balanced accuracy, and MAE of the predicted class midpoint in years."""
    bucket_pred = np.asarray(bucket_pred, dtype=int)
    mae = float(np.abs(YEAR_BUCKET_MIDPOINTS[bucket_pred] - np.asarray(years_true, float)).mean())
    return {**years_class_report(years_true, bucket_pred), "years_mae": mae}


def _grid():
    for values in product(*GRID.values()):
        yield dict(zip(GRID, values))


def tune(group: str) -> dict:
    """Pick grid settings on validation for both targets of one group; one D-19 result row."""
    parts = load_group(group)
    train, val = parts["train"], parts["val"]
    X_tr = build_X(train, group)
    X_va = align_categories(X_tr, build_X(val, group))
    y_tr = log_share(train)
    cap = float(y_tr.max())

    aav = min(((p, report(val["aav_nominal"], predict_aav(fit_aav(X_tr, y_tr, **p), X_va,
                                                            val["offseason"], cap)))
               for p in _grid()), key=lambda r: r[1]["mae_log"])
    years = max(((p, score_years(val["years_new"], fit_years(X_tr, train["years_new"], **p).predict(X_va)))
                 for p in _grid()), key=lambda r: r[1]["years_balanced_accuracy"])

    return {"model": "boosted_tree", "group": group, "n_train": len(X_tr), "n_val": len(X_va),
            **aav[1], **years[1],
            **{f"aav_{k}": v for k, v in aav[0].items()},
            **{f"years_{k}": v for k, v in years[0].items()}}


def main() -> None:
    results = pd.DataFrame([tune(g) for g in GROUPS])[RESULT_COLUMNS]
    out = ROOT / "results/boosted_tree_val_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(results.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
