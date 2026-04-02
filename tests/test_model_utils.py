"""
tests/test_model_utils.py — Unit tests for src/model_utils.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model_utils import evaluate_model, get_feature_importance, get_scale_pos_weight


def _make_imbalanced_target(n_ones: int, n_zeros: int) -> pd.Series:
    return pd.Series([1] * n_ones + [0] * n_zeros)


def _train_minimal_model() -> tuple:
    """Train an XGBClassifier on a tiny synthetic binary dataset."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((50, 3)),
        columns=["feat_a", "feat_b", "feat_c"],
    )
    # ~80% class-1 imbalance
    y = pd.Series([1] * 40 + [0] * 10)
    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=2,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)
    return model, X, y


def test_get_scale_pos_weight_correct_ratio():
    y = _make_imbalanced_target(n_ones=90, n_zeros=10)
    result = get_scale_pos_weight(y)
    assert abs(result - 10 / 90) < 1e-9


def test_get_scale_pos_weight_less_than_one_for_imbalanced():
    # 97% ones (churned majority), 3% zeros (retained minority)
    y = _make_imbalanced_target(n_ones=97, n_zeros=3)
    result = get_scale_pos_weight(y)
    assert result < 1.0


def test_evaluate_model_returns_required_keys():
    model, X, y = _train_minimal_model()
    metrics = evaluate_model(model, X, y)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics, f"Expected key '{key}' missing from evaluate_model result"


def test_get_feature_importance_returns_dataframe():
    model, X, y = _train_minimal_model()
    result = get_feature_importance(model, X.columns.tolist(), X)
    assert isinstance(result, pd.DataFrame)
    for col in ["feature", "mean_abs_shap", "rank"]:
        assert col in result.columns, f"Expected column '{col}' not found"
    assert len(result) == X.shape[1]
    assert list(result["rank"]) == list(range(1, X.shape[1] + 1))
