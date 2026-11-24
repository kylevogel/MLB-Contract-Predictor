"""Evaluation metrics for AAV predictions (DECISIONS.md D-09). Targets are nominal AAV in $M.

Headline: share of contracts predicted within 5% of the actual AAV (the proposal's metric).
Secondary: MAE and R-squared on log AAV, plus MAE in $M, so models stay comparable if 5% is too strict.
Contract length (D-19): every model reports its predictions as the classes 1 / 2 / 3 / 4-5 / 6+ years.
Regressors convert predicted years with `bucket_from_years`; balanced accuracy weights each class equally,
so a model cannot score well by predicting "1 year" for everyone.
"""
import numpy as np


def _arrays(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and the same shape")
    if (y_true <= 0).any() or (y_pred <= 0).any():
        raise ValueError("AAV values must be positive (log metrics)")
    return y_true, y_pred


def within_pct(y_true, y_pred, tol: float = 0.05) -> float:
    """Share of predictions with |pred - true| / true <= tol."""
    y_true, y_pred = _arrays(y_true, y_pred)
    return float((np.abs(y_pred - y_true) / y_true <= tol).mean())


def mae_usd_m(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.abs(y_pred - y_true).mean())


def mae_log(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    return float(np.abs(np.log(y_pred) - np.log(y_true)).mean())


def r2_log(y_true, y_pred) -> float:
    y_true, y_pred = _arrays(y_true, y_pred)
    lt, lp = np.log(y_true), np.log(y_pred)
    ss_tot = ((lt - lt.mean()) ** 2).sum()
    return float(1 - ((lt - lp) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def report(y_true, y_pred) -> dict[str, float]:
    """All AAV metrics in one dict, for logging every baseline and neural-network run the same way."""
    return {
        "within_5pct": within_pct(y_true, y_pred, 0.05),
        "within_10pct": within_pct(y_true, y_pred, 0.10),
        "mae_usd_m": mae_usd_m(y_true, y_pred),
        "mae_log": mae_log(y_true, y_pred),
        "r2_log": r2_log(y_true, y_pred),
    }


YEAR_BUCKET_LABELS = ["1", "2", "3", "4-5", "6+"]


def years_bucket(years) -> np.ndarray:
    """Contract years -> class index: 1 -> 0, 2 -> 1, 3 -> 2, 4-5 -> 3, 6+ -> 4."""
    y = np.asarray(years, dtype=int)
    if (y < 1).any():
        raise ValueError("contract years must be at least 1")
    return np.select([y == 1, y == 2, y == 3, y <= 5], [0, 1, 2, 3], default=4)


def bucket_from_years(pred_years) -> np.ndarray:
    """Predicted (continuous) years -> class index. Rounds half up (2.5 -> 3), never below 1 year."""
    rounded = np.maximum(np.floor(np.asarray(pred_years, dtype=float) + 0.5), 1)
    return years_bucket(rounded)


def years_class_report(years_true, bucket_pred) -> dict[str, float]:
    """Length-class accuracy and balanced accuracy (mean recall over the classes present in years_true)."""
    true_b = years_bucket(years_true)
    pred_b = np.asarray(bucket_pred, dtype=int)
    if true_b.shape != pred_b.shape or true_b.size == 0:
        raise ValueError("years_true and bucket_pred must be non-empty and the same shape")
    recalls = [float((pred_b[true_b == c] == c).mean()) for c in np.unique(true_b)]
    return {"years_bucket_accuracy": float((pred_b == true_b).mean()),
            "years_balanced_accuracy": float(np.mean(recalls))}
