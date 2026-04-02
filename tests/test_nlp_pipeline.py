"""
tests/test_nlp_pipeline.py — Unit tests for src/nlp_pipeline.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_pipeline import add_sentiment_features, score_sentiment


def test_score_sentiment_returns_tuple():
    result = score_sentiment("This product is great!")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)


def test_score_sentiment_handles_none():
    assert score_sentiment(None) == (0.0, 0.0)


def test_score_sentiment_handles_empty_string():
    assert score_sentiment("") == (0.0, 0.0)


def test_add_sentiment_features_adds_columns():
    df = pd.DataFrame({"review_comment_message": ["Good product", "Very fast delivery"]})
    result = add_sentiment_features(df)
    for col in ["has_review_text", "sentiment_polarity", "sentiment_subjectivity"]:
        assert col in result.columns, f"Expected column '{col}' not found"


def test_has_review_text_false_for_null():
    df = pd.DataFrame({"review_comment_message": [None, "Great service"]})
    result = add_sentiment_features(df)
    assert result["has_review_text"].iloc[0] == False
    assert result["has_review_text"].iloc[1] == True
