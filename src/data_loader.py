"""
data_loader.py — Single source of truth for loading and joining the 9 Olist CSV files.

This module is the entry point for all data pipelines in the ecommerce-churn-nlp project.
It loads the Brazilian E-Commerce dataset from Kaggle, joins 9 relational tables into
one order-level DataFrame, applies a right-censoring filter, and attaches the churn label.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------------
# File name constants — single place to update if Kaggle renames files
# ---------------------------------------------------------------------------
_CSV_FILES = {
    "orders":               "olist_orders_dataset.csv",
    "order_items":          "olist_order_items_dataset.csv",
    "order_reviews":        "olist_order_reviews_dataset.csv",
    "customers":            "olist_customers_dataset.csv",
    "payments":             "olist_order_payments_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "geolocation":          "olist_geolocation_dataset.csv",
}

_DATETIME_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_tables(data_dir: str) -> dict[str, pd.DataFrame]:
    """Load all 9 Olist CSV files from data_dir and return them as a dict.

    Each table plays a distinct role in understanding churn for the Olist
    growth team:
      - orders: order lifecycle and timestamps — the spine of the analysis.
      - order_items: what was bought, at what price, with what freight cost.
      - order_reviews: customer-written feedback and star ratings — primary NLP input.
      - customers: maps order-scoped customer_id to customer_unique_id so we can
        count how many times the same person actually shopped.
      - payments: total spend, payment method, installment count per order.
      - products: maps product_id to Portuguese category name.
      - category_translation: translates Portuguese category names to English for
        readability in charts and LLM summaries.
      - sellers: seller geography — useful for future seller-quality analysis.
      - geolocation: lat/lng lookup by zip code — useful for delivery-time modelling.

    Datetime columns in the orders table are parsed to pandas Timestamps so
    downstream code can compute delivery delays and observation windows without
    additional conversion.

    Parameters
    ----------
    data_dir:
        Path to the directory containing the raw Olist CSVs. Can be relative
        (e.g. "data/raw/") or absolute.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'orders', 'order_items', 'order_reviews', 'customers', 'payments',
        'products', 'category_translation', 'sellers', 'geolocation'.

    Raises
    ------
    FileNotFoundError
        If any expected CSV is missing from data_dir, with a message naming the
        missing file so the user can resolve it quickly.
    """
    data_path = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}

    for key, filename in _CSV_FILES.items():
        filepath = data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing required data file: {filepath}\n"
                f"Download the Olist dataset from Kaggle and place all CSVs in '{data_dir}'.\n"
                f"Expected file: {filename}"
            )
        tables[key] = pd.read_csv(filepath)

    # Parse datetime columns in orders so downstream code gets Timestamps, not strings
    for col in _DATETIME_COLS:
        tables["orders"][col] = pd.to_datetime(tables["orders"][col])

    # Parse review timestamp so deduplication in join_tables is chronologically correct
    tables["order_reviews"]["review_answer_timestamp"] = pd.to_datetime(
        tables["order_reviews"]["review_answer_timestamp"]
    )

    return tables


def join_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the 9 Olist tables into a single order-level analytical DataFrame.

    Join strategy
    -------------
    Olist's schema is order-centric: each order can have multiple items and
    multiple payment rows, but for churn modelling we need one row per order.
    We therefore aggregate before joining:

      1. Items aggregation (order_items → products → category_translation):
         Resolves product_id → English category name, then collapses to one row
         per order: item_count, total_item_price, total_freight, and the modal
         product category.

      2. Payments aggregation:
         Olist supports split payments (e.g. voucher + credit card on the same
         order). We sum payment_value, take max installments, and record the
         primary payment type.

      3. Reviews deduplication:
         Olist allows customers to re-submit reviews. We keep the most recent
         review per order by review_answer_timestamp.

      4. Master join sequence:
         orders → customers → order_reviews (deduped) → items_agg → payments_agg

         All joins are left joins from the orders spine so no orders are dropped
         even if they lack reviews, items data, or payment records.

    The result is one row per order, ready for the observation-window filter and
    churn label in subsequent pipeline steps.

    Parameters
    ----------
    tables:
        Dictionary returned by load_raw_tables().

    Returns
    -------
    pd.DataFrame
        One row per order with columns from all joined tables.
    """
    orders      = tables["orders"]
    order_items = tables["order_items"]
    reviews     = tables["order_reviews"]
    customers   = tables["customers"]
    payments    = tables["payments"]
    products    = tables["products"]
    cat_trans   = tables["category_translation"]

    # ------------------------------------------------------------------
    # 1. Aggregate items to order level
    # ------------------------------------------------------------------
    items_with_cat = (
        order_items
        .merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(cat_trans, on="product_category_name", how="left")
    )

    items_agg = (
        items_with_cat
        .groupby("order_id")
        .agg(
            item_count=("order_item_id", "count"),
            total_item_price=("price", "sum"),
            total_freight=("freight_value", "sum"),
            # Modal English category per order; NaN when no category data available
            product_category_english=(
                "product_category_name_english",
                lambda x: x.mode().iloc[0] if x.notna().any() else np.nan,
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 2. Aggregate payments to order level
    # ------------------------------------------------------------------
    payments_agg = (
        payments
        .groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", lambda x: x.iloc[0]),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 3. Deduplicate reviews: keep the most recent review per order
    # ------------------------------------------------------------------
    reviews_deduped = (
        reviews
        .sort_values("review_answer_timestamp")
        .drop_duplicates(subset="order_id", keep="last")
    )

    # ------------------------------------------------------------------
    # 4. Master join
    # ------------------------------------------------------------------
    df = (
        orders
        .merge(
            customers[["customer_id", "customer_unique_id", "customer_state", "customer_city"]],
            on="customer_id",
            how="left",
        )
        .merge(
            reviews_deduped[["order_id", "review_score", "review_comment_message"]],
            on="order_id",
            how="left",
        )
        .merge(items_agg, on="order_id", how="left")
        .merge(payments_agg, on="order_id", how="left")
    )

    return df


def apply_observation_window(df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Filter orders to an observation window ending months before the last order.

    Why this filter is necessary
    ----------------------------
    Churn is defined as a customer placing exactly one order and never returning.
    Without a cutoff, customers who ordered near the end of the dataset look
    churned simply because they haven't had time to place a second order — this
    is called right-censoring bias.

    The 6-month default is chosen because the median inter-purchase gap for
    returning Olist customers is approximately 5–6 months. Any customer included
    in the filtered set has had at least 6 months of opportunity to return; if
    they haven't, labelling them churned is defensible.

    Adjusting `months` shifts the trade-off between label accuracy (higher is
    safer) and data volume (higher loses more recent orders).

    Parameters
    ----------
    df:
        Order-level DataFrame with an order_purchase_timestamp column.
    months:
        Number of months before the last observed order to set as the cutoff.
        Orders on or after this cutoff are excluded.

    Returns
    -------
    pd.DataFrame
        Subset of df containing only orders placed before the cutoff date.
    """
    last_order_date = df["order_purchase_timestamp"].max()
    cutoff = last_order_date - relativedelta(months=months)

    print(f"  Last order in dataset : {last_order_date.date()}")
    print(f"  Observation cutoff    : {cutoff.date()}  ({months}-month window)")

    return df[df["order_purchase_timestamp"] < cutoff].copy()


def add_churn_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a boolean churned column: True if a customer placed exactly one order.

    Churn definition
    ----------------
    A customer is considered churned if customer_unique_id appears exactly once
    in the dataset — i.e. they placed one order and never returned to Olist.

    This is a marketplace-specific definition, not a subscription churn model.
    Limitations:
      - We cannot observe whether the customer switched to a competing marketplace
        (Mercado Livre, Amazon.com.br) — we can only see Olist behaviour.
      - High-value customers who placed one very large order are treated the same
        as low-value one-time buyers; consider weighting by payment_value for
        ROI-driven retention campaigns.
      - customer_unique_id (not customer_id) must be used: Olist creates a new
        customer_id for every order, so customer_id would make every customer
        look like a one-time buyer.

    The helper column orders_placed (count of distinct order_ids per
    customer_unique_id) is also added; it is used for SHAP interpretation and
    should be dropped before model training to avoid label leakage.

    Parameters
    ----------
    df:
        Order-level DataFrame with customer_unique_id and order_id columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns: orders_placed (int) and
        churned (bool).
    """
    order_counts = (
        df.groupby("customer_unique_id")["order_id"]
        .nunique()
        .rename("orders_placed")
    )
    df = df.merge(order_counts, on="customer_unique_id", how="left")
    df["churned"] = df["orders_placed"] == 1
    return df


def load_olist_data(
    data_dir: str,
    observation_window_months: int = 6,
) -> pd.DataFrame:
    """Master pipeline: load raw CSVs → join → filter → label → return.

    This is the single function that notebooks and downstream src/ modules should
    call. It returns an order-level DataFrame that is analysis-ready: all 9 tables
    joined, datetime columns parsed, observation window applied to prevent
    right-censoring bias, and a churned label attached.

    Who calls this
    --------------
    - 02_preprocessing.ipynb: uses the output to engineer features and build the
      train/test split for the XGBoost classifier.
    - 03_modeling.ipynb: imports via feature_engineering.py for consistency.
    - 04_business_impact.ipynb: uses churn rate statistics for the ROI model and
      as context for the Anthropic API executive summary prompts.
    - The __main__ block below: smoke-tests the full pipeline from the command line.

    Progress is printed at each step so callers can confirm the pipeline is
    running correctly and identify where data loss occurs (e.g. failed joins,
    observation window trim).

    Parameters
    ----------
    data_dir:
        Path to the directory containing the raw Olist CSVs.
    observation_window_months:
        Months before the last order to set as the observation cutoff.
        Default 6 matches the analysis in notebooks 01 and 02.

    Returns
    -------
    pd.DataFrame
        One row per order with joined features, orders_placed, and churned columns.
    """
    print("=== load_olist_data ===")

    # Step 1: Load raw tables
    print("\n[1/4] Loading raw CSV files...")
    tables = load_raw_tables(data_dir)
    for key, tbl in tables.items():
        print(f"  {key:22s}: {tbl.shape}")

    # Step 2: Join tables
    print("\n[2/4] Joining tables...")
    df = join_tables(tables)
    print(f"  Post-join shape       : {df.shape}")
    print(f"  Unique customers      : {df['customer_unique_id'].nunique():,}")

    # Step 3: Apply observation window
    print(f"\n[3/4] Applying {observation_window_months}-month observation window...")
    df = apply_observation_window(df, months=observation_window_months)
    print(f"  Post-filter shape     : {df.shape}")
    print(f"  Unique customers      : {df['customer_unique_id'].nunique():,}")

    # Step 4: Add churn label
    print("\n[4/4] Adding churn label...")
    df = add_churn_label(df)

    # Report churn rate at customer level (the correct business metric)
    customer_churn = df.drop_duplicates("customer_unique_id")["churned"]
    churn_rate = customer_churn.mean()
    churned_n  = customer_churn.sum()
    retained_n = (~customer_churn).sum()
    print(f"  Churn rate (customer-level) : {churn_rate:.1%}")
    print(f"    Churned  : {churned_n:,}")
    print(f"    Retained : {retained_n:,}")

    print(f"\nFinal DataFrame shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/"
    df = load_olist_data(data_dir)

    print("\n--- Column names ---")
    for col in df.columns:
        print(f"  {col}")

    print(f"\nShape      : {df.shape}")
    churn_rate = df.drop_duplicates("customer_unique_id")["churned"].mean()
    print(f"Churn rate : {churn_rate:.1%}  (customer-level)")
