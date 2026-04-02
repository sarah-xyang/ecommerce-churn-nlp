"""
tests/test_feature_engineering.py — Unit tests for src/feature_engineering.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from feature_engineering import (
    add_behavioral_features,
    get_feature_columns,
    impute_missing_values,
)


def _make_order_row(
    delivered_offset_days: float = 0.0,
    estimated_offset_days: float = 0.0,
    purchase_offset_days: float = -10.0,
    payment_value: float = 100.0,
    total_freight: float = 10.0,
    item_count: int = 1,
) -> pd.DataFrame:
    """Build a single-row order DataFrame for behavioral feature tests."""
    base = pd.Timestamp("2022-06-01")
    return pd.DataFrame(
        {
            "order_purchase_timestamp": [base + pd.Timedelta(days=purchase_offset_days)],
            "order_estimated_delivery_date": [base + pd.Timedelta(days=estimated_offset_days)],
            "order_delivered_customer_date": [base + pd.Timedelta(days=delivered_offset_days)],
            "payment_value": [payment_value],
            "total_freight": [total_freight],
            "item_count": [item_count],
        }
    )


def test_add_behavioral_features_creates_expected_columns():
    df = _make_order_row()
    result = add_behavioral_features(df)
    for col in ["delivery_delay_days", "is_late_delivery", "days_to_delivery",
                "freight_ratio", "is_single_item"]:
        assert col in result.columns, f"Expected column '{col}' not found"


def test_delivery_delay_positive_when_late():
    # delivered 5 days after estimated
    df = _make_order_row(delivered_offset_days=5, estimated_offset_days=0)
    result = add_behavioral_features(df)
    assert result["delivery_delay_days"].iloc[0] > 0
    assert result["is_late_delivery"].iloc[0] == True


def test_delivery_delay_negative_when_early():
    # delivered 3 days before estimated
    df = _make_order_row(delivered_offset_days=-3, estimated_offset_days=0)
    result = add_behavioral_features(df)
    assert result["delivery_delay_days"].iloc[0] < 0
    assert result["is_late_delivery"].iloc[0] == False


def test_impute_missing_values_no_nulls_after():
    df = pd.DataFrame(
        {
            "delivery_delay_days": [1.0, None, 3.0],
            "review_score": [5.0, None, 4.0],
            "payment_value": [None, 50.0, 100.0],
            "item_count": [1, 1, 2],
        }
    )
    result = impute_missing_values(df)
    for col in ["delivery_delay_days", "review_score", "payment_value"]:
        assert result[col].isna().sum() == 0, f"Nulls remain in '{col}' after imputation"


def test_get_feature_columns_excludes_leaky_columns():
    df = pd.DataFrame(
        {
            "review_score": [4.0],
            "payment_value": [100.0],
            "orders_placed": [1],
            "churned": [True],
        }
    )
    features = get_feature_columns(df)
    assert "orders_placed" not in features
    assert "churned" not in features
