import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from nn_data import GroupData, load_group
from nn_model import ContractMLP
from train_nn import SHARED_COLUMNS, class_weights, evaluate, main, predict, predict_val, train


def _subset(d: GroupData, n: int) -> GroupData:
    """Small copy of a group for fast tests (n train rows, n validation rows)."""
    return GroupData(d.feature_names, d.X_train[:n], d.X_val[:n], d.y_log_aav_train[:n], d.y_log_aav_val[:n],
                     d.y_log_share_train[:n], d.y_log_share_val[:n], d.y_bucket_train[:n], d.y_bucket_val[:n],
                     d.meta_train[:n], d.meta_val[:n], d.mean, d.std)


@pytest.fixture(scope="module")
def small():
    return _subset(load_group("hitters"), 50)


def test_output_shapes():
    log_aav, logits = ContractMLP(26)(torch.zeros(7, 26))
    assert log_aav.shape == (7,) and logits.shape == (7, 5)


def test_class_weights_inverse_frequency():
    w = class_weights(np.array([0, 0, 0, 1]), n_classes=3)
    assert w[0] < w[1] and w[2] == 0 and float(w[:2].mean()) == pytest.approx(1.0)


def test_smoke_two_epochs_fast(small):
    start = time.time()
    model, history = train(small, epochs=2)
    assert time.time() - start < 30 and len(history) == 2
    aav, bucket = predict_val(model, small)
    assert (aav > 0).all() and set(bucket) <= set(range(5))
    assert set(SHARED_COLUMNS[4:]) <= set(evaluate(small, aav, bucket))


def test_cap_at_training_max_share(small):
    model, _ = train(small, epochs=2)
    far_out = np.full((3, small.X_val.shape[1]), 50.0, dtype=np.float32)  # far outside the training data
    with torch.no_grad():
        raw = model(torch.tensor(far_out))[0].numpy()
    cap = float(raw.min()) - 1.0  # below every raw prediction, so it must bind
    aav, _ = predict(model, far_out, [2022, 2022, 2023], cap)
    assert np.allclose(aav, np.exp(cap) * np.array([233, 233, 237]))


def test_same_seed_same_result(small):
    a = evaluate(small, *predict_val(train(small, seed=3, epochs=5)[0], small))
    b = evaluate(small, *predict_val(train(small, seed=3, epochs=5)[0], small))
    assert a == b


def test_main_writes_validation_only(tmp_path):
    for group in ("hitters", "pitchers", "pitchers"):  # the pitcher rerun replaces its row
        main(["--group", group, "--seeds", "2", "--epochs", "3", "--out-dir", str(tmp_path)])
    preds = pd.read_csv(tmp_path / "nn_pitchers_val_predictions.csv")
    assert len(preds) == 2 * 185 and preds["offseason"].between(2022, 2023).all()  # never test classes
    table = pd.read_csv(tmp_path / "nn_val_metrics.csv")
    assert table["group"].tolist() == ["hitters", "pitchers"]
    assert table.columns[:len(SHARED_COLUMNS)].tolist() == SHARED_COLUMNS  # DECISIONS.md D-19
    assert (table["n_seeds"] == 2).all() and "r2_log_std" in table
