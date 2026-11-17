"""Pull FanGraphs season stats (via pybaseball) and build as-of-signing features.

Features use the 3 COMPLETED seasons before the signing date. Because the offseason class
runs Nov 1 (Y) to Oct 31 (Y+1), the last completed season is always the offseason year:
  signing in Dec 2024 (offseason 2024)      -> seasons 2024, 2023, 2022
  extension signed in May 2023 (class 2022) -> seasons 2022, 2021, 2020 (2023 is incomplete)
Using `offseason` (never `signing_date`) matters: free agents have no signing date.

Needs: pip install pybaseball   (run on a machine with normal internet access)
Usage: python src/fetch_stats.py
Output: data/processed/features_hitters.csv, features_pitchers.csv, keyed by KEY_COLS.
"""
import pandas as pd

N_SEASONS = 3
# Unique per contract row (checked on the full dataset; mlb_id + signing_date is NOT unique/available).
KEY_COLS = ["mlb_id", "offseason", "contract_type", "team"]
HITTER_STATS = ["WAR", "PA", "wRC+"]
PITCHER_STATS = ["WAR", "IP", "FIP"]
VOLUME = {"hitter": "PA", "pitcher": "IP"}


def load_season_tables(first: int, last: int):
    """Download FanGraphs batting and pitching tables, one row per player-season."""
    from pybaseball import batting_stats, pitching_stats  # imported here so tests need no network lib

    bat = batting_stats(first, last, qual=1, ind=1)
    pit = pitching_stats(first, last, qual=1, ind=1)
    return bat, pit


def _one_row_per_player_season(table: pd.DataFrame, stats: list[str], volume: str) -> pd.DataFrame:
    """Keep IDfg, Season and the stat columns; if a player-season is repeated, keep the largest volume."""
    t = table[["IDfg", "Season", *stats]].sort_values(volume, ascending=False)
    return t.drop_duplicates(["IDfg", "Season"], keep="first")


def build_features(contracts: pd.DataFrame, table: pd.DataFrame, pos_group: str) -> pd.DataFrame:
    """Return KEY_COLS plus `<stat>_y0.._y{N-1}` (y0 = last completed season) for one position group.

    Missing player-seasons (injured, in the minors, not yet debuted) stay NaN; deciding how to
    mask or fill them is a modeling choice, not a data-cleaning one.
    """
    stats = HITTER_STATS if pos_group == "hitter" else PITCHER_STATS
    seasons = _one_row_per_player_season(table, stats, VOLUME[pos_group])
    base = contracts[contracts.pos_group == pos_group][KEY_COLS + ["fg_id"]].copy()
    out = base[KEY_COLS].copy()
    for k in range(N_SEASONS):
        want = base[["fg_id"]].assign(Season=base.offseason - k)
        got = want.merge(seasons, left_on=["fg_id", "Season"], right_on=["IDfg", "Season"], how="left")
        for col in stats:
            out[f"{col}_y{k}"] = got[col].to_numpy()
    return out.reset_index(drop=True)


def main():
    contracts = pd.read_csv("data/processed/contracts.csv")
    first = int(contracts.offseason.min()) - (N_SEASONS - 1)
    last = int(contracts.offseason.max())
    bat, pit = load_season_tables(first, last)
    build_features(contracts, bat, "hitter").to_csv("data/processed/features_hitters.csv", index=False)
    build_features(contracts, pit, "pitcher").to_csv("data/processed/features_pitchers.csv", index=False)


if __name__ == "__main__":
    main()
