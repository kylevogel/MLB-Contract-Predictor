"""Download the Chadwick Bureau register and build a slim player-ID lookup.

Usage: python src/build_ids.py
Writes data/processed/player_ids.csv
"""
import io
import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"
KEEP = ["key_mlbam", "key_fangraphs", "key_bbref", "name_first", "name_last",
        "birth_year", "birth_month", "birth_day", "mlb_played_first", "mlb_played_last"]


def main(out="data/processed/player_ids.csv"):
    frames = []
    for shard in "0123456789abcdef":
        r = requests.get(URL.format(shard), timeout=120)
        r.raise_for_status()
        frames.append(pd.read_csv(io.StringIO(r.text), usecols=KEEP, low_memory=False))
    ids = pd.concat(frames, ignore_index=True)
    ids = ids[ids.mlb_played_first.notna()]  # keep only people who played in MLB
    ids["full_name"] = ids.name_first.fillna("") + " " + ids.name_last.fillna("")
    ids.to_csv(out, index=False)
    print(f"wrote {len(ids):,} MLB players to {out}")


if __name__ == "__main__":
    main()
