"""Previous-contract features, built only from contracts in this dataset.

For each contract, the previous contract is the same player's contract from the most recent EARLIER
offseason class (strictly earlier, so nothing from the same class can leak). Rookie, pre-arbitration and
arbitration salaries and anything before the 2014 class are not in the dataset, so those players get
has_prev_contract = False and missing values.
"""
import numpy as np
import pandas as pd

PREV_COLUMNS = ["has_prev_contract", "prev_contract_type", "prev_contract_years", "prev_contract_total_m",
                "prev_contract_aav", "prev_contract_aav_adjusted", "years_since_prev_contract"]


def add_previous_contract(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    best = df.sort_values("total_value_nominal").groupby(["mlb_id", "offseason"]).tail(1)  # biggest deal per class
    by_player = {k: g.sort_values("offseason") for k, g in best.groupby("mlb_id")}
    rows = []
    for r in df.itertuples():
        g = by_player.get(r.mlb_id)
        prev = g[g.offseason < r.offseason].tail(1) if g is not None else None
        if prev is None or prev.empty:
            rows.append((False, None, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            p = prev.iloc[0]
            rows.append((True, p.contract_type, p.years_new, p.total_value_nominal, p.aav_nominal,
                         p.aav_adjusted, r.offseason - p.offseason))
    df[PREV_COLUMNS] = pd.DataFrame(rows, index=df.index, columns=PREV_COLUMNS)
    return df
