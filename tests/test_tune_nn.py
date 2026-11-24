import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import pandas as pd
import pytest

pytest.importorskip("torch")
from tune_nn import GRID, pick_best

ROOT = pathlib.Path(__file__).parents[1]


def _row(mae_log, bal, weighted):
    return {"hidden": "[64, 32]", "dropout": 0.3, "weight_decay": 1e-3, "years_weight": 0.5,
            "weighted": weighted, "mae_log": mae_log, "years_balanced_accuracy": bal}


def test_tie_broken_by_balanced_accuracy():
    final = pd.DataFrame([_row(0.581, 0.29, False), _row(0.582, 0.37, True), _row(0.600, 0.50, True)])
    best = pick_best(final)
    assert best["weighted"] is True and best["hidden"] == [64, 32]   # 0.582 is within 0.005; 0.600 is not


def test_clear_winner_kept():
    final = pd.DataFrame([_row(0.570, 0.30, False), _row(0.590, 0.45, True)])
    assert pick_best(final)["weighted"] is False


def test_committed_best_settings_match_tuning_tables():
    best = json.loads((ROOT / "results/nn_best_settings.json").read_text())
    for group in ("hitters", "pitchers"):
        table = pd.read_csv(ROOT / f"results/nn_tuning_{group}.csv")
        assert len(table[table.stage == "grid"]) == 48 and set(best[group]) == set(GRID)
        assert pick_best(table[table.stage == "final"]) == best[group]
