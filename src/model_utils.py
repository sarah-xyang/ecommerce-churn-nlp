"""
model_utils.py — XGBoost training, evaluation, threshold tuning, and SHAP explainability.

This module contains the modelling helpers used by notebook 03. It separates
reusable training and evaluation logic from the notebook so that the same
functions can be called from tests, batch scoring scripts, or future notebooks
without duplicating code.

Pipeline position: called after feature_engineering.py produces X_train/X_test
and before llm_insights.py consumes SHAP-ranked feature drivers for narrative
generation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. Imbalance weight
# ---------------------------------------------------------------------------

def get_scale_pos_weight(y: pd.Series) -> float:
    """Compute XGBoost's scale_pos_weight parameter for a severely imbalanced target.

    What scale_pos_weight does in XGBoost
    --------------------------------------
    XGBoost minimises a loss function that sums gradient contributions across
    all training samples equally by default. Under severe class imbalance, the
    majority-class (churned) samples dominate the loss: the model converges to a
    solution that mostly predicts "churned" because that minimises the raw error
    count. scale_pos_weight corrects this by multiplying every positive-class
    (churned, label=1) gradient contribution by the given weight, amplifying the
    minority class's (retained, label=0) influence on tree splits proportionally.

    Why this dataset needs it
    -------------------------
    Olist's churn rate is approximately 97%: only ~3% of customers placed more
    than one order. A model blind to this imbalance will learn to predict "churned"
    for nearly every customer and still achieve 97% accuracy — completely missing
    the retained customers that are the target of any re-engagement campaign.
    scale_pos_weight = count(label=0) / count(label=1) ≈ 0.03 tells the model
    to penalise errors on the minority (retained) class ~33× more heavily than
    errors on the majority (churned) class, restoring parity in the loss signal
    without any resampling that would alter the training distribution.

    Parameters
    ----------
    y:
        Binary target Series (1 = churned, 0 = retained).

    Returns
    -------
    float
        Ratio of negative-class (retained) count to positive-class (churned) count.
        Will be less than 1.0 when churned is the majority class.
    """
    n_churned = int((y == 1).sum())
    n_retained = int((y == 0).sum())
    return n_retained / n_churned


# ---------------------------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier for churn prediction.

    Hyperparameter rationale
    ------------------------
    n_estimators=300
        Number of boosting rounds (trees). More trees allow the ensemble to
        correct residual errors from earlier rounds. 300 is a good starting
        point for a ~56 k-row, 57-feature dataset — large enough to learn
        complex interactions, small enough to train in seconds.

    max_depth=5
        Maximum depth of each individual tree. Depth-5 trees capture up to
        five-way feature interactions (e.g. "high freight AND late delivery AND
        low review score") while avoiding the extreme memorisation risk of very
        deep trees on a 57-feature matrix. Shallower trees (depth 3–4) would
        underfit; deeper trees (depth 7+) overfit on rare feature combinations.

    learning_rate=0.05
        The fraction by which each new tree's contribution is shrunk before
        being added to the ensemble. A small learning rate requires more trees
        (hence n_estimators=300) but produces a better bias-variance tradeoff:
        each tree corrects errors conservatively rather than making large
        corrections that overshoot the optimum.

    eval_metric='logloss'
        Log-loss (binary cross-entropy) evaluates the quality of the predicted
        probability, not just the predicted class. This is the right metric for
        an imbalanced problem because it rewards well-calibrated probabilities
        across the full range — it does not collapse to accuracy-at-threshold,
        which would be dominated by the majority class. Well-calibrated
        probabilities are also essential for threshold tuning in
        find_optimal_threshold.

    random_state=42
        Seeds XGBoost's internal subsampling for reproducibility across runs.

    Parameters
    ----------
    X_train:
        Feature matrix, numeric, no nulls.
    y_train:
        Binary churn target (1 = churned, 0 = retained).

    Returns
    -------
    xgb.XGBClassifier
        Fitted classifier.
    """
    spw = get_scale_pos_weight(y_train)
    print(f"scale_pos_weight = {spw:.4f}  "
          f"(churned errors weighted {1 / spw:.1f}× more than retained errors)")

    model = xgb.XGBClassifier(
        scale_pos_weight=spw,
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"Training complete — {model.n_estimators} trees, "
          f"{X_train.shape[1]} features, {len(y_train):,} rows")
    return model


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict:
    """Evaluate the churn classifier and report metrics for the retained class.

    Why metrics are reported for label=0 (retained customers)
    ----------------------------------------------------------
    The business question is: which customers are likely to come back and are
    therefore worth targeting with retention spend? A churned customer (label=1)
    who is correctly identified as "will churn" contributes nothing to ROI if we
    cannot change their behaviour. It is the retained class — customers who would
    return — that drives the economic case for any re-engagement campaign.

    Precision (retained): of all customers we flag as likely returners, what
    fraction actually return? Low precision means wasted campaign spend on
    customers who were going to churn regardless.

    Recall (retained): of all customers who would have returned, what fraction
    did we correctly identify for outreach? Low recall means missed revenue
    opportunities — returners who were never contacted.

    F1 (retained): harmonic mean of precision and recall, giving an equal-weight
    single score that the growth team can use to compare models or thresholds
    without separately trading off precision and recall.

    ROC-AUC is reported as a threshold-independent measure of the model's overall
    discriminative power — its ability to rank retained customers above churned
    customers regardless of which decision boundary is applied.

    Parameters
    ----------
    model:
        Fitted XGBClassifier (or any sklearn-compatible classifier with predict_proba).
    X_test:
        Test feature matrix.
    y_test:
        True binary labels (1 = churned, 0 = retained).
    threshold:
        Decision threshold applied to churn probability. Predictions are
        labelled churned (1) if predict_proba[:, 1] >= threshold, retained (0)
        otherwise. Default 0.5.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc, threshold.
        Precision, recall, and f1 are computed for the retained class (pos_label=0).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
        "recall":    recall_score(y_test, y_pred, pos_label=0, zero_division=0),
        "f1":        f1_score(y_test, y_pred, pos_label=0, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_prob),
        "threshold": threshold,
    }

    print(f"\n=== Model Evaluation (threshold={threshold:.2f}) ===")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print(f"  Precision (retained) : {metrics['precision']:.4f}")
    print(f"  Recall    (retained) : {metrics['recall']:.4f}")
    print(f"  F1        (retained) : {metrics['f1']:.4f}")
    print(f"  ROC-AUC              : {metrics['roc_auc']:.4f}")

    return metrics


# ---------------------------------------------------------------------------
# 4. Threshold tuning
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """Find the decision threshold that maximises F1 for the retained class (label=0).

    Business context: the precision-recall tradeoff
    -----------------------------------------------
    The default 0.5 threshold is rarely optimal under severe class imbalance.
    Because only ~3% of customers are retained, the model's raw churn probability
    for a genuine returner may be well below 0.5 — the model "hedges" toward the
    majority class. Lowering the threshold classifies more customers as "retained",
    which:
      - Increases recall: we flag more actual returners for outreach (fewer missed).
      - Decreases precision: we also flag true churners as returners (wasted spend).

    Raising the threshold has the opposite effect — a narrower, higher-confidence
    target list that misses more real returners.

    The F1-optimal threshold balances precision and recall with equal weight.
    It is the right default for exploratory campaign design. The growth team should
    shift the threshold upward if campaign costs are high relative to LTV gain
    (precision matters more), or downward if LTV gain dominates and they can
    absorb some mis-targeting (recall matters more).

    Search strategy
    ---------------
    Thresholds from 0.05 to 0.95 in steps of 0.01 are evaluated. This fine-grained
    grid is inexpensive (91 evaluations) and avoids missing a narrow F1 peak that
    a coarser grid might skip.

    Parameters
    ----------
    model:
        Fitted classifier with a predict_proba method.
    X_test:
        Test feature matrix.
    y_test:
        True binary labels (1 = churned, 0 = retained).

    Returns
    -------
    float
        Threshold value in [0.05, 0.95] that maximises F1 for the retained class.
    """
    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)
            best_precision = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
            best_recall = recall_score(y_test, y_pred, pos_label=0, zero_division=0)

    print(f"\n=== Optimal Threshold (retained class, label=0) ===")
    print(f"  Threshold : {best_threshold:.2f}")
    print(f"  Precision : {best_precision:.4f}")
    print(f"  Recall    : {best_recall:.4f}")
    print(f"  F1        : {best_f1:.4f}")

    return best_threshold


# ---------------------------------------------------------------------------
# 5. SHAP feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value using TreeExplainer.

    Why SHAP over XGBoost's built-in feature importance
    ----------------------------------------------------
    XGBoost's built-in importance measures (gain, weight, cover) are computed
    from tree structure: gain sums the improvement in loss from each split; weight
    counts how often a feature is used to split. Both are biased toward features
    that happen to be placed early in shallow trees, and neither tell you the
    *direction* or *magnitude* of a feature's effect on individual predictions.

    SHAP (SHapley Additive exPlanations) is grounded in cooperative game theory:
    each feature's contribution to a prediction is its fair share of the payoff,
    computed by averaging over all possible orderings of features entering the
    model. This gives three properties that matter for stakeholder communication:

    Consistency: a feature that contributes more to model output always receives
    a higher SHAP value — XGBoost gain is not guaranteed to have this property.

    Additivity: SHAP values sum exactly to the difference between each prediction
    and the dataset base rate, making it possible to verify and audit the
    decomposition.

    Direction: positive SHAP → pushes prediction toward churn; negative SHAP →
    pushes toward retention. This is information the growth team can act on
    directly (e.g. "high freight_ratio pushes customers toward churn → reduce
    freight for first-time buyers").

    TreeExplainer computes exact Shapley values for tree ensembles in polynomial
    time via a tree-traversal algorithm, making it practical on the 14 k-row
    test set without kernel approximations.

    Parameters
    ----------
    model:
        Fitted XGBClassifier.
    feature_names:
        Ordered list of feature column names matching the columns the model was
        trained on (i.e. X_train.columns.tolist()).
    X:
        Feature matrix to compute SHAP values on (typically X_test).

    Returns
    -------
    pd.DataFrame
        Columns: feature (str), mean_abs_shap (float), rank (int).
        Sorted by mean_abs_shap descending; rank starts at 1.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print(f"SHAP values computed — shape: {shap_values.shape}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["rank"] = importance_df.index + 1
    return importance_df[["feature", "mean_abs_shap", "rank"]]


# ---------------------------------------------------------------------------
# Smoke test / standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _src = Path(__file__).parent
    _project_root = _src.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    PROCESSED_DIR = _project_root / "data" / "processed"

    print("=== model_utils smoke test ===\n")

    # Load train/test splits written by 02_preprocessing.ipynb
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    print(f"Loaded splits — X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.1%}")
    print(f"Test  churn rate: {y_test.mean():.1%}\n")

    # Impute any remaining NaNs (mirrors the notebook's in-place fix)
    for col in ["total_freight", "payment_value", "payment_installments"]:
        if col in X_train.columns:
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col]  = X_test[col].fillna(median_val)

    # 1. Train
    model = train_xgboost(X_train, y_train)

    # 2. Evaluate at default threshold
    evaluate_model(model, X_test, y_test, threshold=0.5)

    # 3. Find optimal threshold and re-evaluate
    opt_threshold = find_optimal_threshold(model, X_test, y_test)
    print(f"\nRe-evaluating at optimal threshold ({opt_threshold:.2f}):")
    evaluate_model(model, X_test, y_test, threshold=opt_threshold)

    # 4. SHAP feature importance — top 10
    print("\n=== Top 10 features by mean |SHAP| ===")
    importance_df = get_feature_importance(model, X_train.columns.tolist(), X=X_test)
    print(importance_df.head(10).to_string(index=False))
