import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


DB_PATH = "data/iocc.db"


def init_feedback_table(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT NOT NULL,
            anomaly_timestamp TEXT NOT NULL,
            anomaly_type TEXT NOT NULL,
            ensemble_score REAL NOT NULL,
            predicted_root_cause TEXT NOT NULL,
            corrected_root_cause TEXT,
            accuracy_rating INTEGER NOT NULL,
            actionability_rating INTEGER NOT NULL,
            completeness_rating INTEGER NOT NULL,
            operator_notes TEXT,
            source TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def submit_feedback(
    anomaly_timestamp: str,
    anomaly_type: str,
    ensemble_score: float,
    predicted_root_cause: str,
    accuracy_rating: int,
    actionability_rating: int,
    completeness_rating: int,
    corrected_root_cause: Optional[str] = None,
    operator_notes: Optional[str] = None,
    source: str = "rule_based",
    db_path: str = DB_PATH,
) -> int:
    init_feedback_table(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        INSERT INTO llm_feedback (
            submitted_at, anomaly_timestamp, anomaly_type, ensemble_score,
            predicted_root_cause, corrected_root_cause,
            accuracy_rating, actionability_rating, completeness_rating,
            operator_notes, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now(timezone.utc).isoformat(),
        anomaly_timestamp, anomaly_type, ensemble_score,
        predicted_root_cause, corrected_root_cause,
        accuracy_rating, actionability_rating, completeness_rating,
        operator_notes, source,
    ))
    conn.commit()
    feedback_id = cursor.lastrowid
    conn.close()
    return feedback_id


def get_feedback_summary(db_path: str = DB_PATH) -> dict:
    init_feedback_table(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM llm_feedback").fetchall()
    conn.close()

    if not rows:
        return {"total_ratings": 0, "avg_accuracy": 0, "avg_actionability": 0,
                "avg_completeness": 0, "correction_rate": 0, "ratings": []}

    cols = ["id", "submitted_at", "anomaly_timestamp", "anomaly_type", "ensemble_score",
            "predicted_root_cause", "corrected_root_cause", "accuracy_rating",
            "actionability_rating", "completeness_rating", "operator_notes", "source"]
    records = [dict(zip(cols, row)) for row in rows]

    total = len(records)
    corrections = sum(1 for r in records if r["corrected_root_cause"] and
                      r["corrected_root_cause"] != r["predicted_root_cause"])

    return {
        "total_ratings": total,
        "avg_accuracy": round(sum(r["accuracy_rating"] for r in records) / total, 2),
        "avg_actionability": round(sum(r["actionability_rating"] for r in records) / total, 2),
        "avg_completeness": round(sum(r["completeness_rating"] for r in records) / total, 2),
        "correction_rate": round(corrections / total * 100, 1),
        "ratings": records,
    }


if __name__ == "__main__":
    init_feedback_table()
    fid = submit_feedback(
        anomaly_timestamp="2026-04-06 14:00:00",
        anomaly_type="cascade",
        ensemble_score=0.82,
        predicted_root_cause="cascade_failure",
        accuracy_rating=4,
        actionability_rating=5,
        completeness_rating=4,
        operator_notes="Correct diagnosis, circuit breaker recommendation was spot on.",
        source="rule_based",
    )
    print(f"Feedback submitted with ID: {fid}")
    summary = get_feedback_summary()
    print(f"Summary: {summary}")
