import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path


METRIC_COLS = [
    "cpu_percent",
    "memory_percent",
    "disk_io_mbps",
    "network_throughput_mbps",
    "latency_p95_ms",
    "error_rate",
    "container_restarts",
]

WINDOWS = [4, 12, 60, 240]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour_of_day"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_business_hours"] = ((ts.dt.hour >= 9) & (ts.dt.hour < 18) & (ts.dt.dayofweek < 5)).astype(int)
    df["minutes_since_midnight"] = ts.dt.hour * 60 + ts.dt.minute
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in METRIC_COLS:
        for w in WINDOWS:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)
        df[f"{col}_roll_max_60"] = df[col].rolling(60, min_periods=1).max()
        df[f"{col}_roll_min_60"] = df[col].rolling(60, min_periods=1).min()
    return df


def add_zscore_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in METRIC_COLS:
        mean = df[f"{col}_roll_mean_60"]
        std = df[f"{col}_roll_std_60"].replace(0, 1e-9)
        df[f"{col}_zscore"] = (df[col] - mean) / std
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lag_steps = [4, 16, 60, 240]
    for col in ["cpu_percent", "memory_percent", "latency_p95_ms", "error_rate"]:
        for lag in lag_steps:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag).bfill()
    return df


def add_rate_of_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["cpu_percent", "memory_percent", "latency_p95_ms", "error_rate"]:
        df[f"{col}_delta_1"] = df[col].diff(1).fillna(0)
        df[f"{col}_delta_4"] = df[col].diff(4).fillna(0)
        df[f"{col}_pct_change_4"] = df[col].pct_change(4).replace([np.inf, -np.inf], 0).fillna(0)
    return df


def add_cross_metric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cpu_memory_product"] = df["cpu_percent"] * df["memory_percent"] / 10000
    df["latency_error_product"] = df["latency_p95_ms"] * df["error_rate"]
    df["resource_pressure"] = (df["cpu_percent"] + df["memory_percent"]) / 2
    df["network_disk_ratio"] = df["network_throughput_mbps"] / (df["disk_io_mbps"] + 1e-9)

    combined = pd.concat([df["error_rate"], df["latency_p95_ms"]], axis=1)
    combined.columns = ["error_rate", "latency_p95_ms"]
    corr_vals = (
        combined["error_rate"]
        .rolling(60, min_periods=10)
        .corr(combined["latency_p95_ms"])
        .fillna(0)
    )
    df["error_latency_corr_60"] = corr_vals

    df["high_cpu_high_latency"] = (
        (df["cpu_percent"] > df["cpu_percent_roll_mean_60"] + 2 * df["cpu_percent_roll_std_60"]) &
        (df["latency_p95_ms"] > df["latency_p95_ms_roll_mean_60"] + 2 * df["latency_p95_ms_roll_std_60"])
    ).astype(int)
    return df


def add_anomaly_score_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zscore_cols = [f"{col}_zscore" for col in METRIC_COLS]
    df["composite_zscore"] = df[zscore_cols].abs().mean(axis=1)
    df["max_zscore"] = df[zscore_cols].abs().max(axis=1)
    df["n_metrics_above_2std"] = (df[zscore_cols].abs() > 2).sum(axis=1)
    df["n_metrics_above_3std"] = (df[zscore_cols].abs() > 3).sum(axis=1)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = add_temporal_features(df)
    df = add_rolling_features(df)
    df = add_zscore_features(df)
    df = add_lag_features(df)
    df = add_rate_of_change_features(df)
    df = add_cross_metric_features(df)
    df = add_anomaly_score_features(df)
    return df


def save_features(df: pd.DataFrame, db_path: str = "data/iocc.db") -> None:
    conn = sqlite3.connect(db_path)
    df.to_sql("infrastructure_features", conn, if_exists="replace", index=False)
    conn.close()


def load_features(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM infrastructure_features", conn, parse_dates=["timestamp"])
    conn.close()
    return df


if __name__ == "__main__":
    from src.ingestion.metrics_generator import load_from_sqlite
    print("Loading raw metrics...")
    raw = load_from_sqlite()
    print(f"Raw shape: {raw.shape}")
    print("Engineering features...")
    features = engineer_features(raw)
    save_features(features)
    print(f"Feature matrix shape: {features.shape}")
    print(f"Total columns: {features.shape[1]}")
    print("Saved to data/iocc.db -> infrastructure_features")
