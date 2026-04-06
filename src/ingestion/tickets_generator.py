import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import sqlite3
from pathlib import Path


CATEGORIES = ["infrastructure", "application", "security", "performance", "data", "network"]
PRIORITIES = ["low", "medium", "high", "critical"]
PRIORITY_WEIGHTS = [0.4, 0.35, 0.18, 0.07]
ASSIGNEES = ["alice", "bob", "carol", "dave", "eve", "frank"]
TITLES = {
    "infrastructure": ["Server unresponsive", "Disk space critical", "Container crash loop", "Node failure detected"],
    "application": ["Service returning 500s", "Login failure spike", "API timeout", "Deployment rollback needed"],
    "security": ["Unusual login pattern", "Rate limit breach", "Certificate expiry", "Access anomaly detected"],
    "performance": ["Latency degradation", "Memory leak suspected", "CPU saturation", "DB query slowdown"],
    "data": ["Pipeline failure", "Data quality alert", "Missing records", "Schema mismatch"],
    "network": ["Packet loss detected", "DNS resolution failure", "Load balancer error", "Bandwidth spike"],
}


def generate_support_tickets(
    hours: int = 72,
    base_tickets_per_hour: float = 3.0,
    anomaly_rate: float = 0.03,
    seed: int = 44,
) -> pd.DataFrame:
    np.random.seed(seed)
    start = datetime.now(timezone.utc) - timedelta(hours=hours)
    total_seconds = hours * 3600

    expected_total = int(base_tickets_per_hour * hours)
    offsets = np.sort(np.random.uniform(0, total_seconds, expected_total * 3))
    t_hours = offsets / 3600
    daily_mult = 1 + 0.7 * np.sin(2 * np.pi * t_hours / 24 - np.pi / 2)
    keep = np.random.uniform(0, daily_mult.max(), len(offsets)) < daily_mult
    offsets = offsets[keep][:expected_total]
    total = len(offsets)

    categories = np.random.choice(CATEGORIES, total)
    priorities = np.random.choice(PRIORITIES, total, p=PRIORITY_WEIGHTS)
    assignees = np.random.choice(ASSIGNEES, total)
    resolution_hours = np.random.exponential(4, total)
    titles = [np.random.choice(TITLES[cat]) for cat in categories]

    is_anomaly = np.zeros(total, dtype=int)
    anomaly_type = np.array(["normal"] * total)
    n_anomalies = int(total * anomaly_rate)
    anomaly_indices = np.random.choice(total, n_anomalies, replace=False)

    for idx in anomaly_indices:
        kind = np.random.choice(["ticket_flood", "priority_escalation", "sla_breach"])
        is_anomaly[idx] = 1
        anomaly_type[idx] = kind
        window = min(15, total - idx)
        if kind == "ticket_flood":
            priorities[idx:idx+window] = "high"
        elif kind == "priority_escalation":
            priorities[idx:idx+window] = "critical"
            resolution_hours[idx:idx+window] *= np.random.uniform(3, 8)
        elif kind == "sla_breach":
            resolution_hours[idx:idx+window] *= np.random.uniform(5, 15)

    created_at = [start + timedelta(seconds=float(s)) for s in offsets]
    resolved_at = [
        c + timedelta(hours=float(r)) if np.random.random() > 0.15 else None
        for c, r in zip(created_at, resolution_hours)
    ]
    status = ["resolved" if r is not None else "open" for r in resolved_at]

    df = pd.DataFrame({
        "timestamp": created_at,
        "ticket_id": [f"TKT-{i+1000}" for i in range(total)],
        "title": titles,
        "category": categories,
        "priority": priorities,
        "assignee": assignees,
        "status": status,
        "resolution_hours": resolution_hours.round(2),
        "is_anomaly": is_anomaly,
        "anomaly_type": anomaly_type,
    })
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str = "data/iocc.db") -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql("support_tickets", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM support_tickets", conn, parse_dates=["timestamp"])
    conn.close()
    return df


if __name__ == "__main__":
    print("Generating support tickets...")
    df = generate_support_tickets(hours=72)
    save_to_sqlite(df)
    total = len(df)
    anomalies = int(df["is_anomaly"].sum())
    print(f"Generated {total:,} tickets over 72 hours")
    print(f"Anomalies injected: {anomalies} ({anomalies/total*100:.2f}%)")
    print(f"Priority breakdown: {df['priority'].value_counts().to_dict()}")
    print(f"Categories: {df['category'].nunique()} | Open tickets: {(df['status']=='open').sum()}")
    print("Saved to data/iocc.db -> support_tickets")
