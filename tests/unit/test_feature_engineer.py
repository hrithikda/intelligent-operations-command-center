import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.features.feature_engineer import (
    add_temporal_features,
    add_rolling_features,
    add_zscore_features,
    add_lag_features,
    engineer_features,
    METRIC_COLS,
)


def make_sample_df(n=300):
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    timestamps = [start + timedelta(seconds=i * 15) for i in range(n)]
    np.random.seed(0)
    return pd.DataFrame({
        "timestamp": timestamps,
        "cpu_percent": np.random.uniform(20, 80, n),
        "memory_percent": np.random.uniform(40, 70, n),
        "disk_io_mbps": np.random.uniform(10, 50, n),
        "network_throughput_mbps": np.random.uniform(100, 300, n),
        "latency_p95_ms": np.random.uniform(80, 200, n),
        "error_rate": np.random.uniform(0.001, 0.02, n),
        "container_restarts": np.random.randint(0, 3, n),
        "is_anomaly": np.zeros(n, dtype=int),
        "anomaly_type": ["normal"] * n,
    })


def test_temporal_features_added():
    df = make_sample_df()
    result = add_temporal_features(df)
    for col in ["hour_of_day", "day_of_week", "is_business_hours", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
        assert col in result.columns


def test_temporal_cyclical_range():
    df = make_sample_df()
    result = add_temporal_features(df)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_rolling_features_added():
    df = make_sample_df()
    df = add_temporal_features(df)
    result = add_rolling_features(df)
    assert "cpu_percent_roll_mean_60" in result.columns
    assert "error_rate_roll_std_60" in result.columns


def test_zscore_features_finite():
    df = make_sample_df()
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    result = add_zscore_features(df)
    for col in METRIC_COLS:
        assert np.isfinite(result[f"{col}_zscore"]).all()


def test_lag_features_no_nulls():
    df = make_sample_df()
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_zscore_features(df)
    result = add_lag_features(df)
    assert result["cpu_percent_lag_4"].isna().sum() == 0


def test_engineer_features_shape():
    df = make_sample_df()
    result = engineer_features(df)
    assert result.shape[0] == len(df)
    assert result.shape[1] > 100


def test_engineer_features_no_index_reset():
    df = make_sample_df()
    result = engineer_features(df)
    assert list(result.index) == list(range(len(df)))


def test_composite_zscore_non_negative():
    df = make_sample_df()
    result = engineer_features(df)
    assert (result["composite_zscore"] >= 0).all()


def test_n_metrics_above_2std_integer():
    df = make_sample_df()
    result = engineer_features(df)
    assert result["n_metrics_above_2std"].dtype in [np.int32, np.int64, int]
