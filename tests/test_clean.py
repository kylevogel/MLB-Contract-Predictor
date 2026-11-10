import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pandas as pd
from clean_contracts import clean
from schema import validate, COLUMNS


def row(**kw):
    base = dict(player_name="Test Player", mlb_id=1, fg_id=1, bbref_id="testpl01", offseason=2024,
                signing_date="2024-12-10", contract_type="free_agent", team="NYM", prev_team="SDP",
                is_resign=False, position="RF", pos_group="hitter", age_at_signing=26.0,
                service_years_at_signing=6.0, years_new=15, total_value_nominal=765.0,
                aav_nominal=51.0, aav_adjusted=51.0, has_opt_out=True, has_deferral=True,
                has_incentives=False, source_url="x", notes="",
                is_minor_league=False, is_non_roster=False, is_arbitration=False, is_international=False, is_no_mlb_history=False)
    base.update(kw)
    return base


def test_exclusions_and_validation():
    raw = pd.DataFrame([
        row(),
        row(mlb_id=2, is_arbitration=True),
        row(mlb_id=3, is_minor_league=True),
        row(mlb_id=4, is_international=True),
        row(mlb_id=5, is_non_roster=True),
        row(mlb_id=6, is_no_mlb_history=True),
    ])
    kept, dropped = clean(raw)
    assert len(kept) == 1 and len(dropped) == 5
    assert validate(kept) == []
    assert list(kept.columns) == list(COLUMNS)


def test_bad_rows_flagged():
    raw = pd.DataFrame([row(contract_type="arb_settlement", offseason=2030)])
    kept, _ = clean(raw)
    probs = validate(kept)
    assert any("contract_type" in p for p in probs) and any("outside offseasons" in p for p in probs)


def test_offseason_must_match_date():
    kept, _ = clean(pd.DataFrame([row(offseason=2023, signing_date="2024-12-10")]))
    assert any("class implied" in p for p in validate(kept))
