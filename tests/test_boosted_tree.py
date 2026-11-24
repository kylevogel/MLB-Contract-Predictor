import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
import numpy as np
import pytest
import boosted_tree as bt


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_no_contract_outcome_columns_used(group):
    assert not set(bt.FEATURES[group]) & bt.EXCLUDED
    X = bt.build_X(bt.load_group(group)["train"], group)
    assert list(X.columns) == bt.FEATURES[group]


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_test_classes_never_loaded(group):
    parts = bt.load_group(group)
    assert set(parts) == {"train", "val"}
    assert all(p["offseason"].max() <= 2023 for p in parts.values())


@pytest.mark.parametrize("group", list(bt.GROUPS))
def test_predictions_positive_and_right_length(group):
    parts = bt.load_group(group)
    X_tr = bt.build_X(parts["train"], group)
    X_va = bt.align_categories(X_tr, bt.build_X(parts["val"], group))
    for target in bt.TARGETS:
        y = parts["train"][target].to_numpy(float)
        model = bt.fit(X_tr, np.log(y) if target == "aav_nominal" else y, max_iter=20)
        pred = bt.predict(model, X_va, target)
        assert pred.shape == (len(X_va),)
        assert (pred > 0).all()


def test_score_years():
    m = bt.score_years([1, 2, 6], [1.2, 3.4, 8.0])
    assert m["mae_years"] == pytest.approx((0.2 + 1.4 + 2.0) / 3)
    assert m["exact_year"] == pytest.approx(1 / 3)
    assert m["bucket_acc"] == pytest.approx(2 / 3)  # 6 and 8 are both in the 6+ class
