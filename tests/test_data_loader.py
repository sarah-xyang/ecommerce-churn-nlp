"""
tests/test_data_loader.py — Unit and integration tests for src/data_loader.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    add_churn_label,
    apply_observation_window,
    load_olist_data,
    load_raw_tables,
)

DATA_DIR = "data/raw/"

EXPECTED_KEYS = {
    "orders",
    "order_items",
    "order_reviews",
    "customers",
    "payments",
    "products",
    "category_translation",
    "sellers",
    "geolocation",
}


def test_load_raw_tables_returns_all_keys():
    tables = load_raw_tables(DATA_DIR)
    assert set(tables.keys()) == EXPECTED_KEYS


def test_load_raw_tables_datetime_parsing():
    tables = load_raw_tables(DATA_DIR)
    dtype = tables["orders"]["order_purchase_timestamp"].dtype
    assert dtype == "datetime64[ns]" or str(dtype).startswith("datetime64")


def test_apply_observation_window_reduces_rows():
    dates = pd.date_range("2017-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"order_purchase_timestamp": dates})
    result = apply_observation_window(df, months=6)
    assert len(result) < len(df)


def test_add_churn_label_single_order_is_churned():
    df = pd.DataFrame(
        {
            "customer_unique_id": ["cust_a", "cust_a", "cust_b"],
            "order_id": ["o1", "o2", "o3"],
        }
    )
    result = add_churn_label(df)

    cust_a_churned = result.loc[result["customer_unique_id"] == "cust_a", "churned"].iloc[0]
    cust_b_churned = result.loc[result["customer_unique_id"] == "cust_b", "churned"].iloc[0]

    assert cust_b_churned is True or cust_b_churned == True
    assert cust_a_churned is False or cust_a_churned == False


def test_load_olist_data_churn_rate():
    df = load_olist_data(DATA_DIR)
    customer_churn_rate = df.drop_duplicates("customer_unique_id")["churned"].mean()
    assert 0.95 <= customer_churn_rate <= 0.99
