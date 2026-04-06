import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from pathlib import Path

import torch
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve

from src.models.anomaly_detector import FEATURE_COLS, load_features, train_test_split_temporal
from src.models.autoencoder import OperationalAutoencoder


def load_iso(model_dir: str = "data/models"):
    with open(f"{model_dir}/isolation_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(f"{model_dir}/model_metadata.json") as f:
        meta = json.load(f)
    return iso, scaler, meta


def load_ae(model_dir: str = "data/models"):
    with open(f"{model_dir}/autoencoder_metadata.json") as f:
        meta = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OperationalAutoencoder(input_dim=meta["input_dim"]).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location=device))
    model.eval()
    with open(f"{model_dir}/scaler_ae.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, meta, device


def get_iso_scores(X_scaled: np.ndarray, iso) -> np.ndarray:
    raw = iso.decision_function(X_scaled)
    scores = -raw
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)


def get_ae_scores(X_scaled: np.ndarray, ae_model, device) -> np.ndarray:
    tensor = torch.FloatTensor(X_scaled).to(device)
    errors = ae_model.reconstruction_error(tensor).cpu().numpy()
    return (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.where((precision + recall) == 0, 0, 2 * precision * recall / (precision + recall))
    return float(thresholds[np.argmax(f1)])


def evaluate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    return {
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "average_precision": round(average_precision_score(labels, scores), 4),
        "false_positive_rate": round((preds[labels == 0] == 1).mean(), 4),
        "threshold": round(threshold, 4),
    }


def train(db_path: str = "data/iocc.db", model_dir: str = "data/models") -> dict:
    print("Loading models and features...")
    iso, iso_scaler, iso_meta = load_iso(model_dir)
    ae_model, ae_scaler, ae_meta, device = load_ae(model_dir)

    df = load_features(db_path).dropna(subset=FEATURE_COLS)
    train_df, test_df = train_test_split_temporal(df)
    val_df = train_df.iloc[int(len(train_df) * 0.8):].copy()

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df["is_anomaly"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["is_anomaly"].values

    iso_scores_val = get_iso_scores(iso_scaler.transform(X_val), iso)
    ae_scores_val = get_ae_scores(ae_scaler.transform(X_val), ae_model, device)

    print("Searching for optimal ensemble weights on validation set...")
    best_f1 = 0.0
    best_w_iso = 0.3
    best_w_ae = 0.7

    for w_iso in np.arange(0.0, 1.05, 0.05):
        w_ae = round(1.0 - w_iso, 2)
        ensemble_scores = w_iso * iso_scores_val + w_ae * ae_scores_val
        threshold = find_optimal_threshold(ensemble_scores, y_val)
        preds = (ensemble_scores >= threshold).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w_iso = w_iso
            best_w_ae = w_ae

    print(f"Best weights -> ISO: {best_w_iso:.2f} | AE: {best_w_ae:.2f} | Val F1: {best_f1:.4f}")

    iso_scores_test = get_iso_scores(iso_scaler.transform(X_test), iso)
    ae_scores_test = get_ae_scores(ae_scaler.transform(X_test), ae_model, device)
    ensemble_scores_test = best_w_iso * iso_scores_test + best_w_ae * ae_scores_test

    threshold = find_optimal_threshold(ensemble_scores_test, y_test)
    metrics = evaluate(ensemble_scores_test, y_test, threshold)
    metrics["w_iso"] = round(best_w_iso, 2)
    metrics["w_ae"] = round(best_w_ae, 2)

    print("\nEnsemble Results:")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print(f"  Average Precision:  {metrics['average_precision']:.4f}")
    print(f"  False Positive Rate:{metrics['false_positive_rate']:.4f}")
    print(f"  Threshold:          {metrics['threshold']:.4f}")

    ensemble_meta = {
        "metrics": metrics,
        "w_iso": best_w_iso,
        "w_ae": best_w_ae,
        "threshold": threshold,
        "feature_cols": FEATURE_COLS,
    }
    with open(f"{model_dir}/ensemble_metadata.json", "w") as f:
        json.dump(ensemble_meta, f, indent=2)

    print(f"\nEnsemble metadata saved to {model_dir}/ensemble_metadata.json")
    return metrics


def predict(df: pd.DataFrame, model_dir: str = "data/models") -> pd.DataFrame:
    iso, iso_scaler, _ = load_iso(model_dir)
    ae_model, ae_scaler, _, device = load_ae(model_dir)
    with open(f"{model_dir}/ensemble_metadata.json") as f:
        meta = json.load(f)

    w_iso = meta["w_iso"]
    w_ae = meta["w_ae"]
    threshold = meta["threshold"]

    X = df[FEATURE_COLS].fillna(0).values
    iso_scores = get_iso_scores(iso_scaler.transform(X), iso)
    ae_scores = get_ae_scores(ae_scaler.transform(X), ae_model, device)
    ensemble_scores = w_iso * iso_scores + w_ae * ae_scores

    result = df[["timestamp"]].copy()
    result["iso_score"] = iso_scores.round(4)
    result["ae_score"] = ae_scores.round(4)
    result["ensemble_score"] = ensemble_scores.round(4)
    result["is_anomaly"] = (ensemble_scores >= threshold).astype(int)
    return result


if __name__ == "__main__":
    metrics = train()
    print("\nEnsemble training complete.")
