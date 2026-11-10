"""Pull FanGraphs season stats (via pybaseball) and build as-of-signing features.

Features use the 3 COMPLETED seasons before the signing date:
  signing in Nov-Dec 2024 (offseason 2024)  -> seasons 2022, 2023, 2024
  extension signed in-season, e.g. May 2023 -> seasons 2020..2022 (2023 is incomplete)
This prevents future information leaking into the features.

Needs: pip install pybaseball   (run on a machine with normal internet access)
Usage: python src/fetch_stats.py
"""
import pandas as pd
from pybaseball import batting_stats, pitching_stats

N_SEASONS = 3


def last_completed_season(signing_date: pd.Timestamp) -> int:
    # Seasons end in early Oct; anything signed after Oct 31 has that year complete.
    return signing_date.year if signing_date.month >= 11 else signing_date.year - 1


def load_season_tables(first: int, last: int):
    bat = batting_stats(first, last, qual=1, ind=1)   # ind=1 -> one row per player-season
    pit = pitching_stats(first, last, qual=1, ind=1)
    return bat, pit


def build_features(contracts: pd.DataFrame, bat: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, c in contracts.iterrows():
        end = last_completed_season(c.signing_date)
        feat = {"mlb_id": c.mlb_id, "signing_date": c.signing_date}
        src, cols = (pit, ["WAR", "IP", "FIP"]) if c.pos_group == "pitcher" else (bat, ["WAR", "PA", "wRC+"])
        for k in range(N_SEASONS):
            season = end - k
            hit = src[(src.IDfg == c.fg_id) & (src.Season == season)]
            for col in cols:
                feat[f"{col}_y{k}"] = hit[col].iloc[0] if len(hit) else float("nan")
        rows.append(feat)
    return pd.DataFrame(rows)


def main():
    contracts = pd.read_csv("data/processed/contracts.csv", parse_dates=["signing_date"])
    first = contracts.signing_date.dt.year.min() - N_SEASONS - 1
    last = contracts.signing_date.dt.year.max()
    bat, pit = load_season_tables(first, last)
    hit = contracts[contracts.pos_group != "pitcher"]
    pit_c = contracts[contracts.pos_group == "pitcher"]
    build_features(hit, bat, pit).to_csv("data/processed/features_hitters.csv", index=False)
    build_features(pit_c, bat, pit).to_csv("data/processed/features_pitchers.csv", index=False)


if __name__ == "__main__":
    main()
