import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
import sqlite3
from pathlib import Path


def generate_infrastructure_metrics(
    hours: int = 72,
    interval_seconds: int = 15,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    periods = int(hours * 3600 / interval_seconds)
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    timestamps = [start + timedelta(seconds=i * interval_seconds) for i in range(periods)]

    t = np.arange(periods)
    daily = np.sin(2 * np.pi * t / (86400 / interval_seconds))
    weekly = 0.3 * np.sin(2 * np.pi * t / (7 * 86400 / interval_seconds))
    trend = t / periods * 0.1

    cpu = np.clip(
        40 + 20 * daily + 5 * weekly + trend * 10 + np.random.normal(0, 3, periods), 5, 100
    )
    memory = np.clip(
        55 + 10 * daily + 3 * weekly + trend * 8 + np.random.normal(0, 2, periods), 10, 100
    )
    disk_io = np.clip(
        30 + 15 * daily + np.random.exponential(5, periods), 0, 200
    )
    network_throughput = np.clip(
        200 + 100 * daily + 20 * weekly + np.random.normal(0, 15, periods), 0, 1000
    )
    latency_p95 = np.clip(
        120 + 40 * daily + np.random.exponential(10, periods), 10, 2000
    )
    error_rate = np.clip(
        0.01 + 0.005 * daily + np.random.exponential(0.003, periods), 0, 1
    )
    container_restarts = np.random.poisson(0.1 + 0.05 * (daily + 1), periods)

    is_anomaly = np.zeros(periods, dtype=int)
    anomaly_type = ["normal"] * periods
    n_anomalies = int(periods * anomaly_rate)
    anomaly_indices = np.random.choice(periods, n_anomalies, replace=False)

    for idx in anomaly_indices:
        kind = np.random.choice(["cpu_spike", "memory_leak", "latency_surge", "error_burst", "cascade"])
        is_anomaly[idx] = 1
        anomaly_type[idx] = kind
        window = min(8, periods - idx)

        if kind == "cpu_spike":
            cpu[idx:idx+window] = np.clip(cpu[idx:idx+window] + np.random.uniform(35, 55), 0, 100)
        elif kind == "memory_leak":
            leak = np.linspace(0, np.random.uniform(25, 40), window)
            memory[idx:idx+window] = np.clip(memory[idx:idx+window] + leak, 0, 100)
        elif kind == "latency_surge":
            latency_p95[idx:idx+window] += np.random.uniform(400, 900)
            error_rate[idx:idx+window] = np.clip(error_rate[idx:idx+window] + 0.08, 0, 1)
        elif kind == "error_burst":
            error_rate[idx:idx+window] = np.clip(error_rate[idx:idx+window] + np.random.uniform(0.15, 0.4), 0, 1)
            latency_p95[idx:idx+window] += np.random.uniform(100, 300)
        elif kind == "cascade":
            cpu[idx:idx+window] = np.clip(cpu[idx:idx+window] + 40, 0, 100)
            memory[idx:idx+window] = np.clip(memory[idx:idx+window] + 25, 0, 100)
            latency_p95[idx:idx+window] += 600
            error_rate[idx:idx+window] = np.clip(error_rate[idx:idx+window] + 0.2, 0, 1)
            container_restarts[idx:idx+window] += np.random.randint(3, 8, window)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "cpu_percent": cpu.round(2),
        "memory_percent": memory.round(2),
        "disk_io_mbps": disk_io.round(2),
        "network_throughput_mbps": network_throughput.round(2),
        "latency_p95_ms": latency_p95.round(2),
        "error_rate": error_rate.round(5),
        "container_restarts": container_restarts,
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    })
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = "data/iocc.db") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("infrastructure_metrics", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM infrastructure_metrics", conn, parse_dates=["timestamp"])
    conn.close()
    return df


if __name__ == "__main__":
    print("Generating infrastructure metrics...")
    df = generate_infrastructure_metrics(hours=72, interval_seconds=15, anomaly_rate=0.03)
    save_to_sqlite(df)
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"Generated {total:,} records over 72 hours")
    print(f"Anomalies injected: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Anomaly types: {df[df.is_anomaly==1]['anomaly_type'].value_counts().to_dict()}")
    print("Saved to data/iocc.db")
