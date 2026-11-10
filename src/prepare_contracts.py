"""Spotrac raw scrape -> rule-compliant contracts_raw.csv (input to clean_contracts.py).

Applies the rules in DATA_SPEC.md:
  * class by signing window (Nov 1..Oct 31); keep offseasons 2014..2025 only
  * drop exact duplicates
  * drop minor-league contracts: $0 value, or AAV below that season's MLB minimum salary
  * drop international / no-MLB-history players: no MLB season completed before signing
    (Chadwick `mlb_played_first` > last completed season). Spotrac's `team_from` codes
    (JPN, KOR, MEX, DOM, OS, ...) are used as a cross-check only.
Everything dropped is written to data/processed/prepare_dropped.csv with a reason, and
rows needing a human look go to data/processed/review_*.csv.

Usage: python src/prepare_contracts.py [raw_csv]
"""
import pathlib
import re
import sys
import unicodedata
import pandas as pd
from schema import offseason_of, FIRST_OFFSEASON, LAST_OFFSEASON
from adjusted_aav import attach_adjusted_aav

# MLB minimum salary by season ($M). Verified 2026-11-10 against AP/MLBPA tables (via theScore), Baseball-Reference Bullpen and Spotrac's CBA summary.
MIN_SALARY = {2015: .5075, 2016: .5075, 2017: .535, 2018: .545, 2019: .555, 2020: .5635,
              2021: .5705, 2022: .70, 2023: .72, 2024: .74, 2025: .76, 2026: .78}
FOLLOWUP_FRACTION = 0.5  # same-class FA rows worth < 50% of the player's biggest deal are dropped
PLAYER_EXCLUSIONS = {660271: "two_way_player_ohtani"}  # D-08: Ohtani is excluded entirely (all contracts)
INTL_CODES = {"JPN", "KOR", "MEX", "DOM", "OS", "CUB", "TWN", "AUS", "VEN", "NED", "ITA", ""}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?")


ALIASES = {"michael": "mike", "matthew": "matt", "steven": "steve", "manuel": "manny", "giovanny": "gio",
           "jakob": "jake", "joseph": "joe", "samuel": "sam", "christopher": "chris", "william": "will",
           "nicholas": "nick", "robert": "rob", "jonathan": "jon", "jonathon": "jon", "daniel": "dan",
           "joshua": "josh", "zachary": "zack", "zach": "zack", "anthony": "tony", "edward": "eddie",
           "thomas": "tom", "timothy": "tim", "benjamin": "ben", "alexander": "alex", "nathaniel": "nate",
           "nathan": "nate", "jeffrey": "jeff", "gregory": "greg", "kenneth": "ken", "patrick": "pat",
           "andrew": "andy", "james": "jim", "ronald": "ron", "richard": "rich", "david": "dave"}


def _ascii(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return SUFFIX.sub("", s.replace(".", ""))


def _compact(s: str) -> str:
    return re.sub(r"[^a-z]", "", s)


def name_keys(first: str, last: str) -> set:
    """Match keys that survive initials (A.J.), nicknames (Mike/Michael) and middle initials."""
    f, l = _ascii(first), _ascii(last)
    ftoks = f.split() or [""]
    f1 = _compact(ftoks[0]) if len(ftoks[0]) > 1 or len(ftoks) == 1 else _compact(f)
    return {_compact(ALIASES.get(f1, f1)) + _compact(l), _compact(f) + _compact(l)}


def split_spotrac(name: str):
    toks = _ascii(name).split()
    toks = [toks[0]] + [t for t in toks[1:] if len(t) > 1]  # drop middle initials
    return toks[0], " ".join(toks[1:])


def pos_group(pos: str) -> str:
    p = str(pos).upper()
    if "DH" in p and ("SP" in p or "RP" in p):
        return "two_way"
    return "pitcher" if re.match(r"^(SP|RP|P)\b", p) else "hitter"


def assign_class(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signing_date"] = pd.to_datetime(df["signing_date"], errors="coerce")
    from_date = df.signing_date.map(lambda d: offseason_of(d) if pd.notna(d) else pd.NA)
    df["offseason"] = from_date.where(df.signing_date.notna(), df.spotrac_start_year - 1).astype("Int64")
    return df


def attach_ids(df: pd.DataFrame, ids: pd.DataFrame, overrides: dict | None = None):
    """Join to Chadwick by name keys. A candidate must have an MLB career spanning the signing
    season; ties are broken by age (when Spotrac gives one). Still-ambiguous rows go to review."""
    overrides = overrides or {}
    index: dict = {}
    for i, r in ids.iterrows():
        for k in name_keys(r.name_first if pd.notna(r.name_first) else "", r.name_last if pd.notna(r.name_last) else ""):
            index.setdefault(k, set()).add(i)
    out, ambiguous = [], []
    for _, r in df.iterrows():
        if r.spotrac_id in overrides:
            cands = ids[ids.key_mlbam == overrides[r.spotrac_id]]
        else:
            f, l = split_spotrac(r.player_name)
            idx = set().union(*[index.get(k, set()) for k in name_keys(f, l)])
            cands = ids.loc[sorted(idx)]
            if len(cands) > 1:
                span = cands[(cands.mlb_played_last >= r.offseason) & (cands.mlb_played_first <= r.offseason + 1)]
                cands = span if len(span) >= 1 else cands
            if pd.notna(r.age_at_signing) and len(cands) > 1:
                ok = cands[(cands.birth_year - (r.offseason + 1 - r.age_at_signing)).abs() <= 1]
                cands = ok if len(ok) >= 1 else cands
        if len(cands) == 1:
            c = cands.iloc[0]
            out.append((c.key_mlbam, c.key_fangraphs, c.key_bbref, c.mlb_played_first, c.birth_year, c.birth_month, c.birth_day))
        else:
            out.append((pd.NA,) * 7)
            ambiguous.append({"spotrac_id": r.spotrac_id, "player_name": r.player_name, "position": r.position,
                              "offseason": r.offseason, "n_candidates": len(cands),
                              "candidates": "; ".join(f"{c.key_mlbam}:{c.full_name}(b{int(c.birth_year) if pd.notna(c.birth_year) else '?'},{int(c.mlb_played_first)}-{int(c.mlb_played_last)})" for c in cands.itertuples())})
    cols = ["mlb_id", "fg_id", "bbref_id", "mlb_debut_year", "birth_year", "birth_month", "birth_day"]
    df = df.copy()
    df[cols] = pd.DataFrame(out, index=df.index, columns=cols)
    amb = pd.DataFrame(ambiguous)
    df["n_candidates"] = 1
    if len(amb):
        key = df.spotrac_id.astype(str) + "|" + df.offseason.astype(str)
        df["n_candidates"] = key.map(dict(zip(amb.spotrac_id.astype(str) + "|" + amb.offseason.astype(str), amb.n_candidates))).fillna(1).astype(int)
    return df, amb


def main(raw="data/raw/spotrac_raw_contracts.csv", ids_path="data/processed/player_ids.csv"):
    df = pd.read_csv(raw)
    ids = pd.read_csv(ids_path)
    n0 = len(df)
    dropped = []

    def drop(mask, reason):
        mask = mask.fillna(True).astype(bool)  # an undecidable test never keeps a row silently
        if mask.any():
            d = df[mask].copy(); d["drop_reason"] = reason; dropped.append(d)
        return df[~mask]

    # 1. duplicates
    key = ["spotrac_id", "contract_type", "years_new", "total_value_m", "team_to", "spotrac_start_year"]
    df = drop(df.duplicated(key, keep="first"), "duplicate")
    # 1b. manual corrections where independent sources disagree with Spotrac (data/raw/manual_contract_overrides.csv)
    df["notes"] = ""
    ovp = pathlib.Path("data/raw/manual_contract_overrides.csv")
    if ovp.exists():
        for o in pd.read_csv(ovp).itertuples():
            m = (df.spotrac_id == o.spotrac_id) & (df.years_new == o.orig_years) & ((df.total_value_m - o.orig_total_m).abs() < 0.05)
            df.loc[m, ["years_new", "total_value_m", "aav_nominal_m"]] = [o.years_new, o.total_value_m, o.aav_nominal_m]
            df.loc[m, "notes"] = o.source_note
    # 2. class window
    df = assign_class(df)
    df = drop(df.offseason.isna(), "undated_pre_2013_extension")
    df = drop(~df.offseason.between(FIRST_OFFSEASON, LAST_OFFSEASON).fillna(False).astype(bool), "outside_class_window")
    # 3. invalid terms, minor-league / sub-minimum
    df = drop(df.total_value_m <= 0, "minor_league_zero_value")
    df = drop(df.years_new <= 0, "invalid_zero_years")
    season = df.spotrac_start_year.fillna(df.offseason + 1).astype(int)
    floor = season.map(MIN_SALARY)
    df = drop(df.aav_nominal_m < floor, "below_mlb_minimum_salary")
    # 4. IDs + no-MLB-history / international
    ov = pathlib.Path('data/raw/manual_id_overrides.csv')
    overrides = dict(pd.read_csv(ov)[['spotrac_id', 'mlb_id']].values) if ov.exists() else {}
    df, ambiguous = attach_ids(df, ids, overrides)
    df = drop(df.mlb_id.isna() & (df.n_candidates == 0), "not_in_mlb_register_no_mlb_history")
    df = drop(df.mlb_id.isna(), "ambiguous_player_id")
    df = drop(df.mlb_debut_year > df.offseason, "no_mlb_history_before_signing")
    # 4a. two-way players (D-08)
    df["pos_group"] = df.position.map(pos_group)
    df = drop(df.pos_group == "two_way", "two_way_player")
    for pid, why in PLAYER_EXCLUSIONS.items():
        df = drop(df.mlb_id == pid, why)
    # 4b. follow-up free-agent deals in the same class (released, then re-signed cheaply): keep the biggest
    fa = df.contract_type == "free_agent"
    grp = df[fa].groupby(["mlb_id", "offseason"]).total_value_m
    mx, size = grp.transform("max").reindex(df.index), grp.transform("size").reindex(df.index)
    df = drop(fa & (size > 1) & (df.total_value_m < FOLLOWUP_FRACTION * mx), "followup_deal_same_class")
    # 5. derived columns
    df["is_resign"] = df.team_from == df.team_to
    df["pos_group"] = df.position.map(pos_group)
    sign_ref = df.signing_date.fillna(pd.to_datetime(df.offseason.astype(str) + "-12-15"))
    birth = pd.to_datetime(dict(year=df.birth_year, month=df.birth_month.fillna(7), day=df.birth_day.fillna(1)), errors="coerce")
    df["age_at_signing"] = df.age_at_signing.fillna(((sign_ref - birth).dt.days / 365.25).round(1))
    # review lists
    intl_hint = df[df.team_from.isin(INTL_CODES)]
    intl_hint.to_csv("data/processed/review_intl_code_but_mlb_history.csv", index=False)
    ambiguous.to_csv("data/processed/review_id_ambiguous.csv", index=False)
    multi = df[df.duplicated(["mlb_id", "offseason", "contract_type"], keep=False)]
    multi.to_csv("data/processed/review_same_player_same_class.csv", index=False)
    deferrals = pd.read_csv("data/raw/spotrac_deferrals_raw.csv", sep="|")
    df = attach_adjusted_aav(df, deferrals, sign_ref)  # discounts deferred money (see adjusted_aav.py)
    out = df.rename(columns={"team_to": "team", "team_from": "prev_team", "total_value_m": "total_value_nominal",
                             "aav_nominal_m": "aav_nominal", "spotrac_id": "spotrac_id"})
    season_out = out.spotrac_start_year.fillna(out.offseason + 1).astype(int)
    out["at_league_minimum"] = (out.aav_nominal / season_out.map(MIN_SALARY) - 1).abs() < 0.01
    out["source_url"] = "https://www.spotrac.com/mlb/player/_/id/" + out.spotrac_id.astype(str) + "/" + out.slug
    # overrides change years/total, so match deferrals on the ORIGINAL Spotrac terms for those rows (none have deferrals today)
    for c in ["service_years_at_signing", "has_opt_out", "has_incentives"]:
        out[c] = pd.NA  # not scraped (needs player pages)
    out.to_csv("data/raw/contracts_raw.csv", index=False)
    pd.concat(dropped).to_csv("data/processed/prepare_dropped.csv", index=False)
    summary = pd.concat(dropped).drop_reason.value_counts()
    print(f"in {n0} -> kept {len(out)}\n{summary.to_string()}")
    print(f"review: {len(ambiguous)} ambiguous IDs, {len(intl_hint)} intl-code rows kept, {len(multi)} same-player-same-class rows")


if __name__ == "__main__":
    main()
