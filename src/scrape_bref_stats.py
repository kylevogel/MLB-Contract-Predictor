"""Scrape Baseball-Reference season batting and pitching tables (decision D-15).

Pages: https://www.baseball-reference.com/leagues/majors/{year}-standard-{batting|pitching}.shtml
Output (git-ignored, Sports Reference terms restrict redistribution):
  data/raw/bref_batting.csv, data/raw/bref_pitching.csv  -- one row per player-season.

A player traded mid-season appears once, on his combined row ("2TM", "3TM", ...);
the per-team split rows (class "partial_table") and the league-average row are dropped.
Polite: one request every 6 s (Baseball-Reference asks for at most 20 a minute).

Usage: python src/scrape_bref_stats.py [first_season last_season]   (default 2012 2025)
"""
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

URL = "https://www.baseball-reference.com/leagues/majors/{year}-standard-{kind}.shtml"
DELAY_S = 6
HEADERS = {"User-Agent": "Mozilla/5.0 (course project; contract-predictor scraper)"}

# Baseball-Reference data-stat name -> our column name
BAT_COLS = {"age": "age", "team_name_abbr": "team", "comp_name_abbr": "lg", "b_war": "war",
            "b_games": "g", "b_pa": "pa", "b_r": "r", "b_batting_avg": "ba",
            "b_onbase_plus_slugging_plus": "ops_plus", "pos": "pos", "awards": "awards"}
PIT_COLS = {"age": "age", "team_name_abbr": "team", "comp_name_abbr": "lg", "p_war": "war",
            "p_g": "g", "p_gs": "gs", "p_ip": "ip", "p_earned_run_avg_plus": "era_plus",
            "p_fip": "fip", "p_whip": "whip", "awards": "awards"}
TEXT_COLS = {"team", "lg", "pos", "awards"}


def _find_table(html: str, table_id: str):
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id=table_id)
    if table is None:  # some Sports Reference tables are shipped inside HTML comments
        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            if table_id in c:
                table = BeautifulSoup(c, "lxml").find("table", id=table_id)
                break
    if table is None:
        raise ValueError(f"table {table_id!r} not found")
    return table


def parse_season_table(html: str, kind: str, season: int) -> pd.DataFrame:
    """Parse one season page ('batting' or 'pitching') into one row per player."""
    cols = BAT_COLS if kind == "batting" else PIT_COLS
    table = _find_table(html, f"players_standard_{kind}")
    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if set(tr.get("class") or []) & {"partial_table", "norank", "thead"}:
            continue
        name = tr.find(attrs={"data-stat": "name_display"})
        if name is None or not name.get("data-append-csv"):
            continue
        row = {"bbref_id": name["data-append-csv"], "season": season,
               "name": name.get_text().rstrip("*#+ ").strip()}
        for stat, col in cols.items():
            cell = tr.find(attrs={"data-stat": stat})
            row[col] = cell.get_text().strip() if cell else ""
        rows.append(row)
    df = pd.DataFrame(rows, columns=["bbref_id", "season", "name", *cols.values()])
    for col in df.columns.difference(["bbref_id", "name", *TEXT_COLS]):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.duplicated(["bbref_id", "season"]).any():
        raise ValueError(f"{kind} {season}: duplicate player rows after dropping team splits")
    return df


def fetch(url: str, tries: int = 4) -> str:
    for attempt in range(tries):
        r = requests.get(url, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.text
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(60 * (attempt + 1))  # rate-limited or server error: back off
            continue
        r.raise_for_status()
    raise RuntimeError(f"gave up on {url}")


def main(first: int = 2012, last: int = 2025) -> None:
    for kind in ("batting", "pitching"):
        frames = []
        for year in range(first, last + 1):
            frames.append(parse_season_table(fetch(URL.format(year=year, kind=kind)), kind, year))
            print(f"{kind} {year}: {len(frames[-1])} players", flush=True)
            time.sleep(DELAY_S)
        pd.concat(frames).to_csv(f"data/raw/bref_{kind}.csv", index=False)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:3]))
