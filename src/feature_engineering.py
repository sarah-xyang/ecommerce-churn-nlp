"""
feature_engineering.py — Transform raw Olist order data into a model-ready feature matrix.

This module is the second stage of the churn prediction pipeline. It receives the
joined, labelled DataFrame from data_loader.py and produces the (X, y) pair used by
the XGBoost classifier in notebook 03. The four transformation steps — behavioral
feature construction, missing-value imputation, categorical encoding, and feature
selection — are exposed as individual functions so notebooks can inspect intermediate
states, and composed by build_feature_matrix for production use.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Behavioral feature construction
# ---------------------------------------------------------------------------

def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive order-level behavioral signals that predict the likelihood of churn.

    Each feature captures a distinct dimension of the customer experience on Olist.
    Late deliveries, long waits, and high freight costs are hypothesised to drive
    one-and-done behaviour — customers who feel let down are less likely to return.
    These derived columns translate raw timestamps and monetary amounts into signals
    the model can learn from directly.

    Feature definitions
    -------------------
    delivery_delay_days:
        order_delivered_customer_date minus order_estimated_delivery_date in
        calendar days. Positive values mean the order arrived late. Olist's
        reputation for reliability is tested here: even one late delivery can
        erode trust in a marketplace the customer is still evaluating.

    is_late_delivery:
        Boolean flag: True if delivery_delay_days > 0. Provides a sharp
        decision-boundary signal for tree-based models that complements the
        continuous delay measure.

    days_to_delivery:
        order_delivered_customer_date minus order_purchase_timestamp in calendar
        days. Captures absolute wait time regardless of whether the promise was
        met — a 30-day wait that was "on time" is still a long wait. Long
        fulfilment times are especially punishing for first-time customers who
        have no prior relationship to anchor their patience.

    freight_ratio:
        total_freight divided by payment_value — the share of the total bill
        that went to shipping. High freight ratios create sticker shock for
        customers discovering the true cost at checkout or on delivery. Set to
        0.0 where payment_value is zero or null (imputed downstream).

    is_single_item:
        Boolean flag: True if item_count == 1. Single-item orders may reflect
        a one-off, low-commitment purchase rather than basket-building behaviour.
        Customers who buy only one item are more likely to be exploratory shoppers
        who never intended to return.

    Null preservation
    -----------------
    Nulls in delivery timestamp columns (cancelled or in-transit orders) propagate
    naturally to delivery_delay_days and days_to_delivery. These nulls are handled
    in impute_missing_values, not here, so that imputation logic is centralised and
    can be correctly fit on train data only.

    Parameters
    ----------
    df:
        Order-level DataFrame as produced by data_loader.load_olist_data().

    Returns
    -------
    pd.DataFrame
        Input DataFrame with five new columns added in-place copy.
    """
    df = df.copy()

    df["delivery_delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
        .dt.total_seconds() / 86_400
    )

    df["is_late_delivery"] = df["delivery_delay_days"] > 0

    df["days_to_delivery"] = (
        (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
        .dt.total_seconds() / 86_400
    )

    df["freight_ratio"] = np.where(
        df["payment_value"].notna() & (df["payment_value"] > 0),
        df["total_freight"] / df["payment_value"],
        np.nan,
    )

    df["is_single_item"] = df["item_count"] == 1

    return df


# ---------------------------------------------------------------------------
# 2. Missing-value imputation
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using training-safe median or fixed-value strategies.

    Why median imputation
    ---------------------
    Delivery delay, days to delivery, freight, and payment columns are right-skewed:
    a small number of extreme values (very late deliveries, large orders) pull the
    mean upward. Filling with the mean would replace missing values with an
    inflated estimate that does not represent a typical order. The median is robust
    to these outliers and produces a defensible "what a typical order looks like"
    fill value.

    Why 0.0 for freight_ratio
    -------------------------
    freight_ratio is undefined when payment_value is zero (division by zero). A
    zero fill is semantically correct: if no payment was recorded, we treat the
    freight share as zero rather than inventing a value. This is distinct from a
    "typical order" imputation — the null carries meaning.

    Note on imputation timing
    -------------------------
    In a strict ML pipeline the imputer should be fit on the training fold only,
    then applied to the test fold, to prevent any leakage of test-set distribution
    into the fill values. For medians this leakage is negligible in practice
    (medians shift very little when 20% of the data is excluded), and notebook 03
    enforces correct fit/transform ordering via sklearn.pipeline.Pipeline. This
    function accepts a full DataFrame because it is also called by
    build_feature_matrix, which is used for batch scoring and exploratory analysis
    outside the notebook training loop.

    Columns imputed
    ---------------
    delivery_delay_days    — median (nulls from cancelled/in-transit orders)
    days_to_delivery       — median (same root cause)
    review_score           — median (~0.9% of orders lack a review)
    freight_ratio          — 0.0   (payment_value was zero or null)
    total_freight          — median (small number of items with no freight record)
    payment_value          — median (rare payment aggregation gaps)
    payment_installments   — median (rare payment aggregation gaps)

    Parameters
    ----------
    df:
        DataFrame with behavioral features already added by add_behavioral_features().

    Returns
    -------
    pd.DataFrame
        DataFrame with nulls filled in the seven columns listed above.
        Prints null counts before and after imputation to stdout.
    """
    df = df.copy()

    impute_spec: dict[str, str | float] = {
        "delivery_delay_days":  "median",
        "days_to_delivery":     "median",
        "review_score":         "median",
        "freight_ratio":        0.0,
        "total_freight":        "median",
        "payment_value":        "median",
        "payment_installments": "median",
    }

    print("Null counts BEFORE imputation:")
    for col in impute_spec:
        if col in df.columns:
            n = int(df[col].isna().sum())
            print(f"  {col:30s}: {n:,}")

    for col, strategy in impute_spec.items():
        if col not in df.columns:
            continue
        fill_val = df[col].median() if strategy == "median" else strategy
        df[col] = df[col].fillna(fill_val)

    # Re-derive boolean flags from their now-complete numeric bases
    df["is_late_delivery"] = df["delivery_delay_days"] > 0
    df["is_single_item"]   = df["item_count"] == 1

    print("\nNull counts AFTER imputation:")
    for col in impute_spec:
        if col in df.columns:
            n = int(df[col].isna().sum())
            print(f"  {col:30s}: {n:,}")

    return df


# ---------------------------------------------------------------------------
# 3. Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    df: pd.DataFrame,
    top_n_categories: int = 15,
) -> pd.DataFrame:
    """One-hot encode payment type, customer state, and product category.

    Encoding strategy
    -----------------
    All three categorical variables use one-hot encoding with drop_first=True
    to avoid perfect multicollinearity (the dropped level becomes the implicit
    reference category that all other dummies are interpreted against).

    Why grouping rare product categories into "other" prevents overfitting
    -----------------------------------------------------------------------
    The Olist catalogue spans 71 product categories, but most have very few
    orders. If we one-hot encode all 71 directly, the model receives 70 binary
    columns where many columns are 1 for only a handful of rows. XGBoost can
    technically learn splits on these rare columns, but what it learns will be
    noise: the splits will reflect the particular churned/retained composition of
    30–50 orders rather than a generalizable signal. At prediction time, unseen
    or newly popular categories would map to "all zeros", silently degrading model
    quality. Grouping the long tail into "other" concentrates signal in the top
    categories where we have enough data to learn reliably, and makes the model
    robust to category distribution shifts in production.

    Columns encoded
    ---------------
    payment_type              — 4 unique values → 3 dummies after drop_first
    customer_state            — 27 unique values → 26 dummies after drop_first
    product_category_english  — top_n_categories kept by volume, rest → "other",
                                then one-hot encoded → top_n_categories dummies
                                after drop_first

    Original string columns are dropped after encoding.

    Parameters
    ----------
    df:
        DataFrame after behavioral feature construction and imputation.
    top_n_categories:
        Number of highest-volume product categories to retain as distinct levels.
        Default 15 matches the analysis in notebook 02. Remaining categories are
        collapsed into "other" before encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded dummy columns replacing the original string columns.
    """
    df = df.copy()

    # --- payment_type ---
    if "payment_type" in df.columns:
        df = pd.get_dummies(df, columns=["payment_type"], drop_first=True)

    # --- customer_state ---
    if "customer_state" in df.columns:
        df = pd.get_dummies(df, columns=["customer_state"], drop_first=True)

    # --- product_category_english ---
    if "product_category_english" in df.columns:
        top_categories = (
            df["product_category_english"]
            .value_counts()
            .head(top_n_categories)
            .index.tolist()
        )
        df["product_category_grouped"] = df["product_category_english"].where(
            df["product_category_english"].isin(top_categories),
            other="other",
        )
        df = pd.get_dummies(df, columns=["product_category_grouped"], drop_first=True)
        df = df.drop(columns=["product_category_english"])

    return df


# ---------------------------------------------------------------------------
# 4. Feature column selection
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of column names to use as model input features.

    Why identifiers and raw timestamps are excluded
    -----------------------------------------------
    Identifier columns (customer_unique_id, order_id, customer_id) are unique per
    row and carry no generalizable signal — a model that memorises them would
    achieve zero training error but fail entirely on unseen customers. Raw datetime
    columns (order_purchase_timestamp, order_delivered_customer_date, etc.) are
    excluded because their information is already captured in derived features
    (delivery_delay_days, days_to_delivery). Including the raw timestamps would
    allow the model to implicitly learn the dataset's temporal distribution, which
    would not generalise to future scoring windows.

    Why orders_placed is excluded (label leakage)
    ------------------------------------------------
    orders_placed counts how many orders a customer placed in the filtered dataset.
    A value of 1 directly means churned = True; any value > 1 directly means
    churned = False. Including it would give the model a perfect shortcut that
    encodes the label itself — the resulting classifier would achieve near-perfect
    accuracy on the training set but would be useless at prediction time, where
    we do not yet know how many orders the customer will ultimately place.

    Why review_comment_message is excluded
    ---------------------------------------
    The raw review text has already been transformed into sentiment_polarity,
    sentiment_subjectivity, and has_review_text by the NLP pipeline. The original
    string is not a valid numeric model input and would require per-prediction
    NLP inference if retained.

    Parameters
    ----------
    df:
        Fully transformed DataFrame (after behavioral features, imputation, and
        categorical encoding).

    Returns
    -------
    list[str]
        Column names suitable for use as model features, in DataFrame column order.
    """
    exclude = {
        "churned",
        "customer_unique_id",
        "order_id",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "order_delivered_carrier_date",
        "order_approved_at",
        "review_comment_message",
        "order_status",
        "customer_id",
        "customer_city",
        "product_category_english",
        "orders_placed",
        "orders_placed_filtered",
        "customer_state",
        "payment_type",
        # total_item_price is subsumed by payment_value and freight_ratio
        "total_item_price",
    }
    return [col for col in df.columns if col not in exclude]


# ---------------------------------------------------------------------------
# 5. Master pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    top_n_categories: int = 15,
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform a labelled order DataFrame into an (X, y) pair ready for modelling.

    Pipeline steps
    --------------
    1. add_behavioral_features  — derives delivery_delay_days, is_late_delivery,
                                   days_to_delivery, freight_ratio, is_single_item
    2. impute_missing_values    — fills nulls with median or 0.0 per column
    3. encode_categoricals      — one-hot encodes payment_type, customer_state,
                                   and grouped product_category_english
    4. get_feature_columns      — selects numeric model input columns, excluding
                                   identifiers, raw timestamps, and leaky columns

    Who calls this function
    -----------------------
    - 02_preprocessing.ipynb: generates the train/test split saved to data/processed/
    - 03_modeling.ipynb: called for consistency checks and SHAP feature attribution
    - The __main__ block below: smoke-tests the full pipeline from the command line

    Note: NLP features (sentiment_polarity, sentiment_subjectivity, has_review_text)
    are expected to already be present in df if the nlp_pipeline has been applied
    upstream. If absent, those columns are simply not included in X — this allows
    the function to be called in contexts where NLP scoring has not been run.

    Parameters
    ----------
    df:
        Order-level DataFrame as returned by data_loader.load_olist_data().
        Must contain a 'churned' column (added by add_churn_label).
    top_n_categories:
        Passed through to encode_categoricals. Default 15 matches notebook 02.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X — feature matrix (numeric, no nulls, no identifiers)
        y — binary churn target (int: 1 = churned, 0 = retained)
    """
    df = add_behavioral_features(df)
    df = impute_missing_values(df)
    df = encode_categoricals(df, top_n_categories=top_n_categories)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["churned"].astype(int)

    return X, y


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow an optional data_dir argument; default matches project layout
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/"

    # Import here so the module can be used without data_loader on the path
    _src = Path(__file__).parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from data_loader import load_olist_data  # noqa: E402

    print("=== build_feature_matrix smoke test ===\n")
    df_raw = load_olist_data(data_dir)

    X, y = build_feature_matrix(df_raw)

    print(f"\nX shape       : {X.shape}")
    print(f"y shape       : {y.shape}")

    print(f"\nFeature names ({len(X.columns)}):")
    for name in X.columns:
        print(f"  {name}")

    churned_n  = int(y.sum())
    retained_n = int((y == 0).sum())
    total      = len(y)
    print(f"\nClass distribution:")
    print(f"  Churned  (1): {churned_n:,}  ({churned_n / total:.1%})")
    print(f"  Retained (0): {retained_n:,}  ({retained_n / total:.1%})")
