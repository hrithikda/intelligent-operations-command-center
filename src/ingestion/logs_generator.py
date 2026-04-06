import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path


SERVICES = ["api-gateway", "auth-service", "payment-service", "user-service", "inventory-service", "notification-service"]
ENDPOINTS = ["/health", "/login", "/checkout", "/profile", "/search", "/order", "/callback", "/metrics"]
ERROR_CODES = [200, 200, 200, 200, 201, 400, 401, 404, 429, 500, 502, 503]


def generate_application_logs(
    hours: int = 72,
    total_records: int = 50000,
    anomaly_rate: float = 0.03,
    seed: int = 43,
) -> pd.DataFrame:
    np.random.seed(seed)
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    end = datetime.now(timezone.utc)
    total_seconds = hours * 3600

    offsets = np.sort(np.random.uniform(0, total_seconds, total_records))
    t_hours = offsets / 3600
    daily_mult = 1 + 0.6 * np.sin(2 * np.pi * t_hours / 24 - np.pi / 2)
    keep = np.random.uniform(0, daily_mult.max(), total_records) < daily_mult
    offsets = offsets[keep]
    total = len(offsets)

    timestamps = [start + timedelta(seconds=float(s)) for s in offsets]
    services = np.random.choice(SERVICES, total)
    endpoints = np.random.choice(ENDPOINTS, total)
    status_codes = np.random.choice(ERROR_CODES, total)
    latencies = np.random.lognormal(4.5, 0.5, total)

    is_anomaly = np.zeros(total, dtype=int)
    anomaly_type = np.array(["normal"] * total)

    n_anomalies = int(total * anomaly_rate)
    anomaly_indices = np.random.choice(total, n_anomalies, replace=False)

    for idx in anomaly_indices:
        kind = np.random.choice(["error_storm", "latency_spike", "service_crash"])
        is_anomaly[idx] = 1
        anomaly_type[idx] = kind
        window = min(20, total - idx)
        if kind == "error_storm":
            status_codes[idx:idx+window] = np.random.choice([500, 502, 503], window)
            latencies[idx:idx+window] *= np.random.uniform(2, 5)
        elif kind == "latency_spike":
            latencies[idx:idx+window] *= np.random.uniform(8, 20)
        elif kind == "service_crash":
            status_codes[idx:idx+window] = 503
            latencies[idx:idx+window] = np.random.uniform(5000, 30000, window)

    log_levels = np.where(status_codes >= 500, "ERROR", np.where(status_codes >= 400, "WARN", "INFO"))

    df = pd.DataFrame({
        "timestamp": timestamps,
        "service": services,
        "endpoint": endpoints,
        "status_code": status_codes,
        "log_level": log_levels,
        "latency_ms": latencies.round(2),
        "request_size_bytes": np.random.exponential(512, total).astype(int),
        "response_size_bytes": np.random.exponential(2048, total).astype(int),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    })
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = "data/iocc.db") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("application_logs", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM application_logs", conn, parse_dates=["timestamp"])
    conn.close()
    return df


if __name__ == "__main__":
    print("Generating application logs...")
    df = generate_application_logs(hours=72, total_records=50000)
    save_to_sqlite(df)
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"Generated {total:,} log records over 72 hours")
    print(f"Anomalies injected: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Services: {df['service'].nunique()} | Endpoints: {df['endpoint'].nunique()}")
    print(f"Status distribution: {df['status_code'].value_counts().head(4).to_dict()}")
    print("Saved to data/iocc.db -> application_logs")
