import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pandas as pd
import pytest
from build_features import build_features

# Made-up players and numbers.
CONTRACTS = pd.DataFrame([
    {"bbref_id": "hitta01", "offseason": 2021, "pos_group": "hitter"},
    {"bbref_id": "pitcb01", "offseason": 2023, "pos_group": "pitcher"},
    {"bbref_id": "gonec01", "offseason": 2023, "pos_group": "pitcher"},
])
BATTING = pd.DataFrame([
    # hitta01 (window 2019-2021): 2018 is before it, 2020 gets scaled, 2021 is missing, 2022 is after signing
    {"bbref_id": "hitta01", "season": 2018, "war": 9.0, "r": 120, "ba": .350, "ops_plus": 180, "awards": "MVP-1"},
    {"bbref_id": "hitta01", "season": 2019, "war": 3.0, "r": 90, "ba": .280, "ops_plus": 120, "awards": "MVP-3"},
    {"bbref_id": "hitta01", "season": 2020, "war": 1.0, "r": 30, "ba": .250, "ops_plus": 100, "awards": "GG"},
    {"bbref_id": "hitta01", "season": 2022, "war": 8.0, "r": 110, "ba": .330, "ops_plus": 170, "awards": "MVP-1"},
    # a pitcher's Silver Slugger appears only in the batting table
    {"bbref_id": "pitcb01", "season": 2022, "war": 0.1, "r": 2, "ba": .200, "ops_plus": 40, "awards": "SS"},
])
PITCHING = pd.DataFrame([
    {"bbref_id": "pitcb01", "season": 2021, "war": 2.0, "era_plus": 110, "fip": 3.5, "whip": 1.10, "awards": ""},
    {"bbref_id": "pitcb01", "season": 2023, "war": 4.0, "era_plus": 150, "fip": 2.9, "whip": 0.95, "awards": "CYA-2"},
    {"bbref_id": "gonec01", "season": 2015, "war": 1.0, "era_plus": 100, "fip": 4.0, "whip": 1.30, "awards": ""},
])
BATTING["pa"] = 600      # full seasons, so the playing-time minimum (D-16) does not apply here
PITCHING["ip"] = 150.0


@pytest.fixture
def feats():
    return build_features(CONTRACTS, BATTING, PITCHING)


def test_window_and_2020_scaling(feats):
    h = feats.loc[0]
    assert h.seasons_in_window == 2                                  # 2019 and 2020 only
    assert h.war_3yr == pytest.approx((3.0 + 1.0 * 162 / 60) / 2)    # 2020 WAR scaled
    assert h.r_3yr == pytest.approx((90 + 30 * 162 / 60) / 2)        # 2020 runs scaled
    assert h.ba_3yr == pytest.approx((.280 + .250) / 2)              # rate stats not scaled
    assert h.ops_plus_3yr == pytest.approx(110)


def test_missing_season_spreads_weight(feats):
    p = feats.loc[1]
    assert p.seasons_in_window == 2
    assert p.era_plus_3yr == pytest.approx(130) and p.whip_3yr == pytest.approx(1.025)


def test_awards_winners_only_and_from_both_tables(feats):
    h, p = feats.loc[0], feats.loc[1]
    assert h.won_mvp == 0            # MVP-3 is a placing; the MVP-1 seasons are outside the window
    assert h.won_gold_glove == 1
    assert p.won_cy_young == 0       # CYA-2 does not count
    assert p.won_silver_slugger == 1


def test_no_seasons_in_window(feats):
    g = feats.loc[2]
    assert g.seasons_in_window == 0 and pd.isna(g.war_3yr) and g.won_mvp == 0


def test_tiny_seasons_skipped_unless_all_tiny():
    contracts = pd.DataFrame([
        {"bbref_id": "mixed01", "offseason": 2024, "pos_group": "hitter"},
        {"bbref_id": "small01", "offseason": 2024, "pos_group": "hitter"},
        {"bbref_id": "relie01", "offseason": 2024, "pos_group": "pitcher"},
    ])
    bat = pd.DataFrame([
        {"bbref_id": "mixed01", "season": 2022, "pa": 600, "ba": .250},
        {"bbref_id": "mixed01", "season": 2023, "pa": 550, "ba": .260},
        {"bbref_id": "mixed01", "season": 2024, "pa": 6, "ba": .500},     # tiny: skipped
        {"bbref_id": "small01", "season": 2023, "pa": 31, "ba": .150},    # all tiny: all used
        {"bbref_id": "small01", "season": 2024, "pa": 5, "ba": .000},
    ]).assign(war=1.0, r=10, ops_plus=100, awards="")
    pit = pd.DataFrame([
        {"bbref_id": "relie01", "season": 2023, "ip": 60.0, "era_plus": 120},
        {"bbref_id": "relie01", "season": 2024, "ip": 5.0, "era_plus": 40},  # tiny: skipped
    ]).assign(war=1.0, fip=3.5, whip=1.2, awards="")
    f = build_features(contracts, bat, pit)
    assert f.loc[0].seasons_in_window == 2 and f.loc[0].ba_3yr == pytest.approx(.255)
    assert f.loc[1].seasons_in_window == 2 and f.loc[1].ba_3yr == pytest.approx(.075)
    assert f.loc[2].seasons_in_window == 1 and f.loc[2].era_plus_3yr == pytest.approx(120)
