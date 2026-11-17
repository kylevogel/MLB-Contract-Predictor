"""Evaluation metrics for AAV predictions (DECISIONS.md D-09). Targets are nominal AAV in $M.

Headline: share of contracts predicted within 5% of the actual AAV (the proposal's metric).
Secondary: MAE and R-squared on log AAV, plus MAE in $M, so models stay comparable if 5% is too strict.
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
