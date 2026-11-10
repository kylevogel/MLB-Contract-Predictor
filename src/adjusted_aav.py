"""Adjusted (luxury-tax style) AAV: discount deferred money back to the signing date.

Source of deferrals: Spotrac "Deferred Money -> all-time" list (data/raw/spotrac_deferrals_raw.csv).
Contracts with no deferral have aav_adjusted == aav_nominal. For deferral contracts:
    adjusted_total = total - deferred + sum(deferred_payment_t / (1 + r) ** years_from_signing_to_t)
with equal annual deferred payments on Nov 30 of each deferral year and r = DISCOUNT_RATE.

Why r = 3%: published present values imply 2.9% (MLB's own luxury-tax AAV for Ohtani), 3.2% (MLBPA, Ohtani),
3.6% (Betts), 2.5% (Strasburg), 3.3% (Tucker); Lindor (1.2%) is the outlier. See DECISIONS.md D-12.
This is an estimate, not the official figure.
"""
import numpy as np
import pandas as pd

DISCOUNT_RATE = 0.03


def _year(s: str) -> int:
    n = int(s)
    return n if n >= 1000 else (2000 + n if n < 50 else 1900 + n)


def deferral_pv(deferred: float, def_years: str, signing: pd.Timestamp, rate: float = DISCOUNT_RATE) -> float:
    a, b = (_year(x) for x in def_years.split("-"))
    pay = deferred / (b - a + 1)
    t = np.array([max((pd.Timestamp(year=y, month=11, day=30) - signing).days / 365.25, 0) for y in range(a, b + 1)])
    return float((pay / (1 + rate) ** t).sum())


def attach_adjusted_aav(df: pd.DataFrame, deferrals: pd.DataFrame, signing_ref: pd.Series,
                        rate: float = DISCOUNT_RATE) -> pd.DataFrame:
    """df needs spotrac_id, years_new, total_value_m, aav_nominal_m. signing_ref: a Timestamp per row."""
    d = deferrals.dropna(subset=["deferred"]).copy()
    invalid = d.deferred > d.total  # Spotrac lists some deferrals larger than the contract itself; do not trust them
    d["key"] = d.spotrac_id.astype(str) + "|" + d.years.astype(str) + "|" + (d.total / 1e6).round(1).astype(str)
    lookup = {r.key: (r.deferred / 1e6, r.def_years) for r in d[~invalid].itertuples()}
    bad_keys = set(d[invalid].key)
    df = df.copy()
    keys = df.spotrac_id.astype(str) + "|" + df.years_new.astype(str) + "|" + df.total_value_m.round(1).astype(str)
    adj, deferred_m, method = [], [], []
    for (i, r), k in zip(df.iterrows(), keys):
        if k in lookup:
            dm, dy = lookup[k]
            total_adj = r.total_value_m - dm + deferral_pv(dm * 1e6, dy, signing_ref[i], rate) / 1e6
            adj.append(round(total_adj / r.years_new, 3)); deferred_m.append(dm); method.append(f"pv_estimate_{rate:.0%}")
        else:
            adj.append(r.aav_nominal_m); deferred_m.append(0.0)
            method.append("equal_to_nominal_deferral_data_invalid" if k in bad_keys else "equal_to_nominal_no_deferral")
    df["aav_adjusted"], df["deferred_amount_m"], df["aav_adjusted_method"] = adj, deferred_m, method
    df["has_deferral"] = df.deferred_amount_m > 0
    return df
