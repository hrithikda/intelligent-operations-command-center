import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path


TRANSACTION_TYPES = ["purchase", "refund", "transfer", "subscription", "withdrawal"]
TYPE_WEIGHTS = [0.55, 0.15, 0.15, 0.10, 0.05]
CUSTOMER_SEGMENTS = ["retail", "enterprise", "premium", "trial"]
SEGMENT_WEIGHTS = [0.50, 0.20, 0.20, 0.10]
CURRENCIES = ["USD", "EUR", "GBP", "CAD"]
CURRENCY_WEIGHTS = [0.70, 0.15, 0.10, 0.05]
STATUS_CODES = ["success", "success", "success", "success", "failed", "pending", "declined"]


def generate_transactions(
    hours: int = 72,
    total_records: int = 20000,
    anomaly_rate: float = 0.03,
    seed: int = 45,
) -> pd.DataFrame:
    np.random.seed(seed)
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    total_seconds = hours * 3600

    offsets = np.sort(np.random.uniform(0, total_seconds, total_records))
    t_hours = offsets / 3600
    daily_mult = 1 + 0.5 * np.sin(2 * np.pi * t_hours / 24 - np.pi / 2)
    keep = np.random.uniform(0, daily_mult.max(), total_records) < daily_mult
    offsets = offsets[keep]
    total = len(offsets)

    timestamps = [start + timedelta(seconds=float(s)) for s in offsets]
    tx_types = np.random.choice(TRANSACTION_TYPES, total, p=TYPE_WEIGHTS)
    segments = np.random.choice(CUSTOMER_SEGMENTS, total, p=SEGMENT_WEIGHTS)
    currencies = np.random.choice(CURRENCIES, total, p=CURRENCY_WEIGHTS)
    statuses = np.random.choice(STATUS_CODES, total)

    amounts = np.where(
        tx_types == "enterprise",
        np.random.lognormal(7, 1, total),
        np.where(tx_types == "premium", np.random.lognormal(5, 0.8, total),
        np.random.lognormal(4, 1.2, total))
    )
    processing_ms = np.random.lognormal(5, 0.6, total)

    is_anomaly = np.zeros(total, dtype=int)
    anomaly_type = np.array(["normal"] * total)
    n_anomalies = int(total * anomaly_rate)
    anomaly_indices = np.random.choice(total, n_anomalies, replace=False)

    for idx in anomaly_indices:
        kind = np.random.choice(["fraud_pattern", "volume_spike", "processing_slowdown", "failure_surge"])
        is_anomaly[idx] = 1
        anomaly_type[idx] = kind
        window = min(20, total - idx)
        if kind == "fraud_pattern":
            amounts[idx:idx+window] *= np.random.uniform(10, 50)
            statuses[idx:idx+window] = "declined"
        elif kind == "volume_spike":
            amounts[idx:idx+window] *= np.random.uniform(5, 20)
        elif kind == "processing_slowdown":
            processing_ms[idx:idx+window] *= np.random.uniform(10, 30)
            statuses[idx:idx+window] = "pending"
        elif kind == "failure_surge":
            statuses[idx:idx+window] = "failed"
            processing_ms[idx:idx+window] *= np.random.uniform(2, 5)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "transaction_id": [f"TXN-{i+10000}" for i in range(total)],
        "transaction_type": tx_types,
        "customer_segment": segments,
        "currency": currencies,
        "amount": amounts.round(2),
        "status": statuses,
        "processing_time_ms": processing_ms.round(2),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    })
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = "data/iocc.db") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM transactions", conn, parse_dates=["timestamp"])
    conn.close()
    return df


if __name__ == "__main__":
    print("Generating transactions...")
    df = generate_transactions(hours=72, total_records=20000)
    save_to_sqlite(df)
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"Generated {total:,} transactions over 72 hours")
    print(f"Anomalies injected: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Type breakdown: {df['transaction_type'].value_counts().to_dict()}")
    print(f"Status breakdown: {df['status'].value_counts().to_dict()}")
    print(f"Avg amount: ${df['amount'].mean():.2f} | Max: ${df['amount'].max():.2f}")
    print("Saved to data/iocc.db -> transactions")
