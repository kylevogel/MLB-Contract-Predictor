"""D-14 regression: each real-world contract appears once per player."""
import pandas as pd

# Verified distinct contracts that share a signature inside one class window (see DECISIONS.md D-14):
# Ichiro 2014: FA 1y/$2M for 2015, then an Oct-2015 1y/$2M for 2016. Hedges 2024: FA Nov 2024 for 2025, then Oct-2025 for 2026.
REVIEWED_SAME_CLASS_SIGNATURES = {(400085, 2014), (595978, 2024)}


def _df():
    return pd.read_csv("data/processed/contracts.csv")


def test_no_identical_contract_twice_in_a_class():
    d = _df()
    key = ["mlb_id", "offseason", "team", "years_new", "total_value_nominal"]
    dup = d[d.duplicated(key, keep=False)]
    unexpected = {(int(a), int(b)) for a, b in zip(dup.mlb_id, dup.offseason)} - REVIEWED_SAME_CLASS_SIGNATURES
    assert not unexpected, dup[dup.set_index(["mlb_id", "offseason"]).index.isin(list(unexpected))]


def test_renegotiation_and_duplicate_rows_dropped():
    d = _df()
    lopez = d[(d.player_name == "Reynaldo López") & (d.total_value_nominal == 30.0)]
    assert len(lopez) == 1 and lopez.iloc[0].contract_type == "free_agent"
    moose = d[(d.player_name == "Mike Moustakas") & (d.offseason == 2017)]
    assert len(moose) == 1


def test_superseded_qualifying_offer_dropped():
    d = _df()
    assert not ((d.player_name == "Jose Abreu") & (d.offseason == 2019) & (d.contract_type == "free_agent")).any()
