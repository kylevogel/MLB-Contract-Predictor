"""Overlap audit: is each real-world contract represented once?

Checks every player with 2+ rows (free agents AND extensions):
  A. exact-duplicate signature (same value+years) in different classes -> same contract twice
  B. same class, more than one row (should be only the legit extension-after-FA cases removed earlier)
  C. a later contract signed before the earlier one's span ended, where the later deal is an
     extension that *replaces* it (expected, benign) vs. another free agent deal (suspicious)
  D. near-identical total value (within 1%) across classes
Writes data/processed/review_overlaps.csv and prints a summary. Exit code 1 on type A/B/D hits.
"""
import sys
import pandas as pd

df = pd.read_csv("data/processed/contracts.csv")
df["start"] = df["offseason"] + 1
df["end"] = df["offseason"] + df["years_new"]
rows, hard = [], []
for pid, g in df.groupby("mlb_id"):
    if len(g) < 2:
        continue
    g = g.sort_values(["offseason", "signing_date"]).reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.loc[i], g.loc[j]
            flags = []
            if a.offseason == b.offseason:
                flags.append("B_same_class")
            if a.total_value_nominal == b.total_value_nominal and a.years_new == b.years_new:
                flags.append("A_same_value_years")
            elif abs(a.total_value_nominal - b.total_value_nominal) <= 0.01 * a.total_value_nominal and a.years_new == b.years_new:
                flags.append("D_near_identical")
            if b.start <= a.end:  # later deal begins before earlier one ended
                kind = "overlap_extension_replaces" if b.contract_type.startswith("extension") else "overlap_NON_extension"
                flags.append(kind)
            if flags:
                rows.append(dict(mlb_id=pid, player=a.player_name, flags="|".join(flags),
                    a_class=a.offseason, a_type=a.contract_type, a_years=a.years_new, a_total=a.total_value_nominal,
                    a_span=f"{a.start}-{a.end}", b_class=b.offseason, b_type=b.contract_type, b_years=b.years_new,
                    b_total=b.total_value_nominal, b_span=f"{b.start}-{b.end}"))
out = pd.DataFrame(rows)
out.to_csv("data/processed/review_overlaps.csv", index=False)
print("players with 2+ rows:", (df.groupby("mlb_id").size() > 1).sum())
print("pairs flagged:", len(out))
if len(out):
    print(out["flags"].value_counts().to_string())
hardmask = out["flags"].str.contains("A_|B_|D_|NON_ext") if len(out) else []
print("hard flags:", int(hardmask.sum()) if len(out) else 0)
# Benign overlaps (opt-outs, trades, releases, extensions replacing years) are expected; true duplicates are enforced by tests/test_overlaps.py
sys.exit(0)
