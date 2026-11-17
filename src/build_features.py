"""Build D-15 performance features and merge them onto the contracts.

Inputs : data/processed/contracts.csv, data/raw/bref_batting.csv, data/raw/bref_pitching.csv
Outputs: data/processed/contracts_features_hitters.csv, data/processed/contracts_features_pitchers.csv
         data/processed/features_dropped.csv   (contracts with no MLB season in the window, with reason)

Window: the 3 completed seasons before signing. For every contract the last completed season
equals its offseason class (D-05), so the window is offseason-2 .. offseason.
Stats are equal-weight averages over the seasons the player appears in (missing seasons are skipped).
Seasons under MIN_PLAYING_TIME (50 PA / 10 IP) are skipped, unless every window season is under it (D-16).
2020 counting stats (WAR, R) are scaled to 162 games first. Awards: 1 if the player WON the award
(Baseball-Reference marks MVP-1 / CYA-1 / GG / SS) in any window season, in either table.

Usage: python src/build_features.py
"""
import pandas as pd

WINDOW = 3
SCALE_2020 = 162 / 60
STATS = {"hitter": ["war", "r", "ba", "ops_plus"], "pitcher": ["war", "era_plus", "fip", "whip"]}
COUNTING = ["war", "r"]
MIN_PLAYING_TIME = {"hitter": ("pa", 50), "pitcher": ("ip", 10)}
AWARDS = {"won_mvp": "MVP-1", "won_cy_young": "CYA-1", "won_gold_glove": "GG", "won_silver_slugger": "SS"}


def scale_2020(seasons: pd.DataFrame) -> pd.DataFrame:
    """Scale counting stats in the shortened 2020 season to a 162-game pace."""
    out = seasons.copy()
    cols = [c for c in COUNTING if c in out.columns]
    out[cols] = out[cols].astype(float)  # runs are stored as ints; scaled values are not whole
    out.loc[out.season == 2020, cols] = out.loc[out.season == 2020, cols] * SCALE_2020
    return out


def award_flags(awards: pd.Series) -> pd.DataFrame:
    """One 0/1 column per award, from strings like 'AS,MVP-1,SS'. Placing in a vote does not count."""
    tokens = awards.fillna("").str.split(",")
    return pd.DataFrame({col: tokens.apply(lambda t, code=code: int(code in t)) for col, code in AWARDS.items()},
                        index=awards.index)


def window_rows(contracts: pd.DataFrame, seasons: pd.DataFrame) -> pd.DataFrame:
    """Pair each contract (by row index) with its player's seasons in offseason-2 .. offseason."""
    c = contracts[["bbref_id", "offseason"]].reset_index(names="row")
    m = c.merge(seasons, on="bbref_id")
    return m[(m.season <= m.offseason) & (m.season > m.offseason - WINDOW)]


def drop_tiny_seasons(w: pd.DataFrame, col: str, minimum: float) -> pd.DataFrame:
    """Drop seasons under the playing-time minimum, unless all of a contract's seasons are under it."""
    big = w[col] >= minimum
    return w[big | ~big.groupby(w.row).transform("any")]


def build_features(contracts: pd.DataFrame, batting: pd.DataFrame, pitching: pd.DataFrame) -> pd.DataFrame:
    """One row per contract (same index as `contracts`): averaged stats, seasons_in_window, award flags."""
    awards = pd.concat([batting, pitching])[["bbref_id", "season", "awards"]]
    awards = pd.concat([awards[["bbref_id", "season"]], award_flags(awards.awards)], axis=1)
    award_w = window_rows(contracts, awards).groupby("row")[list(AWARDS)].max()

    parts = []
    for group, table in (("hitter", batting), ("pitcher", pitching)):
        sub = contracts[(contracts.pos_group == "pitcher") == (group == "pitcher")]
        col, minimum = MIN_PLAYING_TIME[group]
        w = window_rows(sub, scale_2020(table[["bbref_id", "season", col, *STATS[group]]]))
        w = drop_tiny_seasons(w, col, minimum)
        f = w.groupby("row")[STATS[group]].mean().add_suffix("_3yr")
        f["seasons_in_window"] = w.groupby("row").season.nunique()
        parts.append(f.reindex(sub.index))
    feats = pd.concat(parts).join(award_w)
    feats["seasons_in_window"] = feats.seasons_in_window.fillna(0).astype(int)
    feats[list(AWARDS)] = feats[list(AWARDS)].fillna(0).astype(int)
    return feats.loc[contracts.index]


def main() -> None:
    contracts = pd.read_csv("data/processed/contracts.csv")
    batting = pd.read_csv("data/raw/bref_batting.csv")
    pitching = pd.read_csv("data/raw/bref_pitching.csv")
    merged = contracts.join(build_features(contracts, batting, pitching))

    no_seasons = merged.seasons_in_window == 0
    dropped = merged.loc[no_seasons, ["player_name", "bbref_id", "offseason", "contract_type", "pos_group"]].copy()
    dropped["reason"] = "no MLB season in the 3-season window before signing (D-15)"
    dropped.to_csv("data/processed/features_dropped.csv", index=False)
    merged = merged[~no_seasons]

    for group in ("hitter", "pitcher"):
        rows = merged[(merged.pos_group == "pitcher") == (group == "pitcher")]
        other = STATS["pitcher" if group == "hitter" else "hitter"]
        drop_cols = [f"{s}_3yr" for s in other if s not in STATS[group]]
        rows.drop(columns=drop_cols).to_csv(f"data/processed/contracts_features_{group}s.csv", index=False)
        print(f"{group}s: {len(rows)} rows")
    print(f"dropped: {len(dropped)} rows -> data/processed/features_dropped.csv")


if __name__ == "__main__":
    main()
