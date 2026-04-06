import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.metrics import precision_recall_curve

from src.models.anomaly_detector import FEATURE_COLS, load_features, train_test_split_temporal


class OperationalAutoencoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            recon = self.forward(x)
            return torch.mean((x - recon) ** 2, dim=1)


def find_threshold_from_labels(
    errors: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    scores_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
    precision, recall, thresholds = precision_recall_curve(labels, scores_norm)
    f1_scores = np.where(
        (precision + recall) == 0, 0, 2 * precision * recall / (precision + recall)
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(precision[best_idx]), float(recall[best_idx])


def train(db_path: str = "data/iocc.db", model_dir: str = "data/models") -> dict:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading features...")
    df = load_features(db_path)
    df = df.dropna(subset=FEATURE_COLS)

    train_df, test_df = train_test_split_temporal(df)
    normal_train = train_df[train_df["is_anomaly"] == 0]
    print(f"Normal training samples: {len(normal_train):,}")
    print(f"Test samples: {len(test_df):,} | Anomalies: {int(test_df['is_anomaly'].sum())}")

    scaler_ae = StandardScaler()
    X_train = scaler_ae.fit_transform(normal_train[FEATURE_COLS].values)
    X_test = scaler_ae.transform(test_df[FEATURE_COLS].values)
    y_test = test_df["is_anomaly"].values

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)

    dataset = TensorDataset(X_train_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = OperationalAutoencoder(input_dim=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    print("Training autoencoder...")
    best_loss = float("inf")
    best_state = None
    epochs = 60

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    errors = model.reconstruction_error(X_test_tensor).cpu().numpy()
    errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)

    threshold, _, _ = find_threshold_from_labels(errors, y_test)
    predictions = (errors_norm >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    ap = average_precision_score(y_test, errors_norm)
    fpr = (predictions[y_test == 0] == 1).mean()

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(ap, 4),
        "false_positive_rate": round(fpr, 4),
        "threshold": round(threshold, 4),
        "best_train_loss": round(best_loss, 6),
    }

    print("\nAutoencoder Results:")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1 Score:           {f1:.4f}")
    print(f"  Average Precision:  {ap:.4f}")
    print(f"  False Positive Rate:{fpr:.4f}")
    print(f"  Threshold:          {threshold:.4f}")

    torch.save(model.state_dict(), f"{model_dir}/autoencoder.pt")
    with open(f"{model_dir}/scaler_ae.pkl", "wb") as f:
        pickle.dump(scaler_ae, f)
    with open(f"{model_dir}/autoencoder_metadata.json", "w") as f:
        json.dump({"metrics": metrics, "input_dim": len(FEATURE_COLS), "threshold": threshold}, f, indent=2)

    print(f"\nAutoencoder saved to {model_dir}/")
    return metrics


def load_model(model_dir: str = "data/models") -> Tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f"{model_dir}/autoencoder_metadata.json") as f:
        meta = json.load(f)
    model = OperationalAutoencoder(input_dim=meta["input_dim"]).to(device)
    model.load_state_dict(torch.load(f"{model_dir}/autoencoder.pt", map_location=device))
    model.eval()
    with open(f"{model_dir}/scaler_ae.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler, meta, device


def predict(df: pd.DataFrame, model_dir: str = "data/models") -> pd.DataFrame:
    model, scaler, meta, device = load_model(model_dir)
    threshold = meta["threshold"]
    X = scaler.transform(df[FEATURE_COLS].fillna(0).values)
    X_tensor = torch.FloatTensor(X).to(device)
    errors = model.reconstruction_error(X_tensor).cpu().numpy()
    errors_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-9)
    result = df[["timestamp"]].copy()
    result["ae_reconstruction_error"] = errors.round(6)
    result["ae_anomaly_score"] = errors_norm.round(4)
    result["ae_is_anomaly"] = (errors_norm >= threshold).astype(int)
    return result


if __name__ == "__main__":
    metrics = train()
    print("\nTraining complete.")
