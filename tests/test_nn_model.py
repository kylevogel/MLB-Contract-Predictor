import sys, pathlib, time, json
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from nn_data import GroupData, load_group
from nn_model import ContractMLP
from train_nn import train, predict, evaluate, main


def _subset(d: GroupData, n: int) -> GroupData:
    """Small copy of a group for fast tests (n train rows, n validation rows)."""
    return GroupData(d.feature_names, d.X_train[:n], d.X_val[:n], d.y_log_aav_train[:n], d.y_log_aav_val[:n],
                     d.y_bucket_train[:n], d.y_bucket_val[:n], d.meta_train[:n], d.meta_val[:n], d.mean, d.std)


@pytest.fixture(scope="module")
def small():
    return _subset(load_group("hitters"), 50)


def test_output_shapes():
    log_aav, logits = ContractMLP(23)(torch.zeros(7, 23))
    assert log_aav.shape == (7,) and logits.shape == (7, 5)


def test_smoke_two_epochs_fast(small):
    start = time.time()
    model, history = train(small, epochs=2)
    assert time.time() - start < 10 and len(history) == 2
    aav, bucket = predict(model, small.X_val)
    assert (aav > 0).all() and set(bucket) <= set(range(5))
    m = evaluate(small, aav, bucket)
    assert {"within_5pct", "mae_usd_m", "r2_log", "years_bucket_accuracy", "years_mae"} <= set(m)


def test_same_seed_same_result(small):
    a = evaluate(small, *predict(train(small, seed=3, epochs=5)[0], small.X_val))
    b = evaluate(small, *predict(train(small, seed=3, epochs=5)[0], small.X_val))
    assert a == b


def test_main_writes_validation_only(tmp_path):
    main(["--group", "pitchers", "--epochs", "3", "--out-dir", str(tmp_path)])
    preds = pd.read_csv(tmp_path / "nn_pitchers_val_predictions.csv")
    assert len(preds) == 185 and preds["offseason"].between(2022, 2023).all()  # never test classes
    main(["--group", "hitters", "--epochs", "3", "--out-dir", str(tmp_path)])
    main(["--group", "pitchers", "--epochs", "3", "--out-dir", str(tmp_path)])  # rerun replaces its row
    table = pd.read_csv(tmp_path / "nn_val_metrics.csv")
    assert table["group"].tolist() == ["hitters", "pitchers"]
    assert table.set_index("group").loc["pitchers", "n_val"] == 185
    shared = ["model", "group", "n_train", "n_val", "within_5pct", "within_10pct", "mae_usd_m",
              "mae_log", "r2_log", "years_bucket_accuracy", "years_mae"]  # DECISIONS.md D-18
    assert table.columns[:len(shared)].tolist() == shared
