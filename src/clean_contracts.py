"""Turn the hand/scraped raw contract table into the validated dataset.

Input : data/raw/contracts_raw.csv  (columns: see schema.COLUMNS, plus the
        exclusion flags is_minor_league, is_non_roster, is_arbitration,
        is_international, is_no_mlb_history)
Output: data/processed/contracts.csv
Usage : python src/clean_contracts.py
"""
import sys
import pandas as pd
from schema import COLUMNS, validate
from prev_contract import add_previous_contract

EXCLUSION_FLAGS = ["is_minor_league", "is_non_roster", "is_arbitration", "is_international",
                   "is_no_mlb_history"]


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, dropped). `dropped` records why each row was removed."""
    raw = raw.copy()
    for f in EXCLUSION_FLAGS:
        raw[f] = raw.get(f, False)
        raw[f] = raw[f].fillna(False).astype(bool)
    drop_mask = raw[EXCLUSION_FLAGS].any(axis=1)
    dropped = raw[drop_mask].copy()
    dropped["drop_reason"] = dropped[EXCLUSION_FLAGS].apply(
        lambda r: ",".join(c for c in EXCLUSION_FLAGS if r[c]), axis=1)
    kept = raw[~drop_mask].drop(columns=EXCLUSION_FLAGS)
    kept["signing_date"] = pd.to_datetime(kept["signing_date"])
    kept["aav_nominal"] = (kept.total_value_nominal / kept.years_new).round(3)
    kept["aav_adjusted"] = kept.aav_adjusted.fillna(kept.aav_nominal)
    kept = add_previous_contract(kept)
    for c in COLUMNS:  # optional columns not scraped yet stay empty
        if c not in kept.columns:
            kept[c] = pd.NA
    return kept[list(COLUMNS)], dropped


def main(src="data/raw/contracts_raw.csv", out="data/processed/contracts.csv"):
    kept, dropped = clean(pd.read_csv(src))
    dropped.to_csv("data/processed/contracts_dropped.csv", index=False)
    problems = validate(kept)
    if problems:
        print("VALIDATION PROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    kept.to_csv(out, index=False)
    print(f"kept {len(kept)} rows, dropped {len(dropped)} (see contracts_dropped.csv)")


if __name__ == "__main__":
    main()
