import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
from fetch_stats import build_features, KEY_COLS, N_SEASONS

ROOT = pathlib.Path(__file__).parents[1]


def _contracts():
    return pd.DataFrame({
        "mlb_id": [1, 2, 3], "fg_id": [10, 20, 30], "offseason": [2024, 2022, 2024],
        "contract_type": ["free_agent", "extension_veteran", "free_agent"], "team": ["NYM", "ATL", "LAD"],
        "pos_group": ["hitter", "hitter", "pitcher"],
    })


def _bat():
    return pd.DataFrame({
        "IDfg": [10, 10, 10, 10, 20, 20],
        "Season": [2024, 2023, 2022, 2025, 2022, 2020],   # 2025 must never be used for offseason 2024
        "WAR": [5.0, 4.0, 3.0, 9.9, 2.0, 1.0], "PA": [600, 580, 550, 100, 500, 60],
        "wRC+": [150, 140, 130, 999, 110, 90],
    })


def test_hitter_features_use_only_completed_seasons():
    f = build_features(_contracts(), _bat(), "hitter")
    r = f[f.mlb_id == 1].iloc[0]
    assert (r.WAR_y0, r.WAR_y1, r.WAR_y2) == (5.0, 4.0, 3.0)      # 2024, 2023, 2022
    assert 9.9 not in f.filter(like="WAR").to_numpy()               # 2025 (future) is never used


def test_missing_seasons_are_nan_and_rows_not_lost():
    f = build_features(_contracts(), _bat(), "hitter")
    r = f[f.mlb_id == 2].iloc[0]                                    # offseason 2022 -> 2022, 2021, 2020
    assert r.WAR_y0 == 2.0 and np.isnan(r.WAR_y1) and r.WAR_y2 == 1.0
    assert len(f) == 2                                              # pitcher row not in hitter table
    assert list(f.columns[: len(KEY_COLS)]) == KEY_COLS


def test_free_agents_without_signing_date_work():
    # the old implementation read signing_date, which is empty for every free agent
    assert "signing_date" not in _contracts().columns
    build_features(_contracts(), _bat(), "hitter")


def test_pitcher_columns_and_duplicate_player_seasons():
    pit = pd.DataFrame({"IDfg": [30, 30], "Season": [2024, 2024], "WAR": [1.0, 3.0], "IP": [10, 150], "FIP": [5.0, 3.5]})
    f = build_features(_contracts(), pit, "pitcher")
    assert f.shape[0] == 1 and f.WAR_y0.iloc[0] == 3.0 and f.FIP_y0.iloc[0] == 3.5   # keeps the larger-IP row
    assert f.filter(like="_y").shape[1] == 3 * N_SEASONS


def test_last_completed_season_equals_offseason_on_real_data():
    df = pd.read_csv(ROOT / "data/processed/contracts.csv", parse_dates=["signing_date"])
    dated = df.dropna(subset=["signing_date"])
    last_completed = np.where(dated.signing_date.dt.month >= 11, dated.signing_date.dt.year, dated.signing_date.dt.year - 1)
    assert (last_completed == dated.offseason).all()
    assert not df.duplicated(KEY_COLS).any()                        # the feature-join key is unique
