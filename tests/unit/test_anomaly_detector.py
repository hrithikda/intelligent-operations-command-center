import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.models.anomaly_detector import (
    find_optimal_threshold,
    train_test_split_temporal,
    FEATURE_COLS,
)


def test_train_test_split_ratio():
    df = pd.DataFrame({"a": range(100)})
    train, test = train_test_split_temporal(df, test_ratio=0.2)
    assert len(train) == 80
    assert len(test) == 20


def test_train_test_split_no_overlap():
    df = pd.DataFrame({"a": range(100)})
    train, test = train_test_split_temporal(df)
    assert len(train) + len(test) == len(df)


def test_find_optimal_threshold_returns_float():
    scores = np.random.uniform(0, 1, 1000)
    labels = (scores > 0.7).astype(int)
    threshold, precision, recall = find_optimal_threshold(scores, labels)
    assert isinstance(threshold, float)
    assert 0 <= threshold <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1


def test_find_optimal_threshold_high_anomaly_score():
    scores = np.array([0.1] * 900 + [0.9] * 100)
    labels = np.array([0] * 900 + [1] * 100)
    threshold, precision, recall = find_optimal_threshold(scores, labels)
    assert threshold > 0.5


def test_feature_cols_not_empty():
    assert len(FEATURE_COLS) > 30


def test_feature_cols_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
