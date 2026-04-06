import numpy as np
import pandas as pd
import sqlite3
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve
import pickle


FEATURE_COLS = [
    "cpu_percent", "memory_percent", "disk_io_mbps", "network_throughput_mbps",
    "latency_p95_ms", "error_rate", "container_restarts",
    "cpu_percent_roll_mean_60", "cpu_percent_roll_std_60",
    "memory_percent_roll_mean_60", "memory_percent_roll_std_60",
    "latency_p95_ms_roll_mean_60", "latency_p95_ms_roll_std_60",
    "error_rate_roll_mean_60", "error_rate_roll_std_60",
    "cpu_percent_zscore", "memory_percent_zscore",
    "latency_p95_ms_zscore", "error_rate_zscore",
    "container_restarts_zscore", "disk_io_mbps_zscore",
    "network_throughput_mbps_zscore",
    "cpu_percent_delta_1", "cpu_percent_delta_4",
    "memory_percent_delta_1", "latency_p95_ms_delta_1",
    "error_rate_delta_1", "error_rate_pct_change_4",
    "cpu_memory_product", "latency_error_product",
    "resource_pressure", "error_latency_corr_60",
    "high_cpu_high_latency", "composite_zscore", "max_zscore",
    "n_metrics_above_2std", "n_metrics_above_3std",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_business_hours",
]


def load_features(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM infrastructure_features", conn, parse_dates=["timestamp"])
    conn.close()
    return df


def train_test_split_temporal(df: pd.DataFrame, test_ratio: float = 0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = np.where(
        (precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def train(db_path: str = "data/iocc.db", model_dir: str = "data/models") -> dict:
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print("Loading features...")
    df = load_features(db_path)
    df = df.dropna(subset=FEATURE_COLS)

    train_df, test_df = train_test_split_temporal(df)
    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"Train anomaly rate: {train_df['is_anomaly'].mean():.3f}")
    print(f"Test anomaly rate:  {test_df['is_anomaly'].mean():.3f}")

    X_train = train_df[FEATURE_COLS].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_anomaly"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=0.03,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)

    raw_scores = iso.decision_function(X_test_scaled)
    anomaly_scores = -raw_scores
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())

    threshold, best_precision, best_recall = find_optimal_threshold(anomaly_scores, y_test)
    predictions = (anomaly_scores >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    ap = average_precision_score(y_test, anomaly_scores)
    fpr = (predictions[y_test == 0] == 1).mean()

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
        "false_positive_rate": round(fpr, 4),
        "threshold": round(threshold, 4),
        "test_samples": len(y_test),
        "test_anomalies": int(y_test.sum()),
    }

    print("\nIsolation Forest Results:")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Average Precision:  {ap:.4f}")
    print(f"  False Positive Rate:{fpr:.4f}")
    print(f"  Threshold:          {threshold:.4f}")

    with open(f"{model_dir}/isolation_forest.pkl", "wb") as f:
        pickle.dump(iso, f)
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{model_dir}/model_metadata.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "feature_cols": FEATURE_COLS,
            "threshold": threshold,
        }, f, indent=2)

    print(f"\nModel saved to {model_dir}/")
    return metrics


def predict(
    df: pd.DataFrame,
    model_dir: str = "data/models",
) -> pd.DataFrame:
    with open(f"{model_dir}/isolation_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/model_metadata.json") as f:
        meta = json.load(f)

    threshold = meta["threshold"]
    X = df[FEATURE_COLS].fillna(0).values
    X_scaled = scaler.transform(X)

    raw_scores = iso.decision_function(X_scaled)
    anomaly_scores = -raw_scores
    anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-9)

    result = df[["timestamp"]].copy()
    result["anomaly_score"] = anomaly_scores.round(4)
    result["is_predicted_anomaly"] = (anomaly_scores >= threshold).astype(int)
    return result


if __name__ == "__main__":
    metrics = train()
    print("\nTraining complete.")
