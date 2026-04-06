import json
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.features.feature_engineer import engineer_features, load_features, save_features
from src.ingestion.metrics_generator import load_from_sqlite
from src.models.ensemble import predict as ensemble_predict
from src.models.forecaster import build_forecast_features, get_feature_cols, load_metrics
from src.reasoning.rag_engine import analyze_anomaly

DB_PATH = "data/iocc.db"
MODEL_DIR = "data/models"
CHROMA_PATH = "data/chroma"

app = FastAPI(
    title="IOCC API",
    description="Intelligent Operations Command Center — prediction, anomaly, and reasoning endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnomalyAnalysisRequest(BaseModel):
    timestamp: str
    anomaly_type: str
    ensemble_score: float
    cpu_percent: float
    memory_percent: float
    latency_p95_ms: float
    error_rate: float
    container_restarts: int


class HealthResponse(BaseModel):
    status: str
    version: str
    db_available: bool
    models_available: bool


def db_conn():
    return sqlite3.connect(DB_PATH)


def models_available() -> bool:
    required = ["ensemble_metadata.json", "autoencoder.pt", "isolation_forest.pkl", "forecaster.pkl"]
    return all(Path(f"{MODEL_DIR}/{f}").exists() for f in required)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "db_available": Path(DB_PATH).exists(),
        "models_available": models_available(),
    }


@app.get("/api/v1/anomalies", tags=["Anomalies"])
def get_anomalies(
    hours: int = Query(24, ge=1, le=72, description="Lookback window in hours"),
    threshold: float = Query(0.12, ge=0.05, le=0.95, description="Anomaly score threshold"),
    limit: int = Query(50, ge=1, le=500),
):
    try:
        features = load_features(DB_PATH)
        preds = ensemble_predict(features, MODEL_DIR)
        preds = preds.rename(columns={"is_anomaly": "predicted_anomaly"})

        raw = load_from_sqlite(DB_PATH)[["timestamp", "cpu_percent", "memory_percent",
                                         "latency_p95_ms", "error_rate", "container_restarts",
                                         "is_anomaly", "anomaly_type"]]
        raw = raw.rename(columns={"is_anomaly": "ground_truth_anomaly"})
        df = preds.merge(raw, on="timestamp", how="left")

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max() - timedelta(hours=hours)
        window = df[df["timestamp"] >= cutoff]
        flagged = window[window["ensemble_score"] >= threshold].sort_values("ensemble_score", ascending=False)

        records = []
        for _, row in flagged.head(limit).iterrows():
            records.append({
                "timestamp": str(row["timestamp"]),
                "ensemble_score": round(float(row["ensemble_score"]), 4),
                "ae_score": round(float(row["ae_score"]), 4),
                "iso_score": round(float(row["iso_score"]), 4),
                "cpu_percent": round(float(row["cpu_percent"]), 2),
                "memory_percent": round(float(row["memory_percent"]), 2),
                "latency_p95_ms": round(float(row["latency_p95_ms"]), 2),
                "error_rate": round(float(row["error_rate"]), 5),
                "container_restarts": int(row["container_restarts"]),
                "anomaly_type": str(row["anomaly_type"]),
            })

        return {
            "total_anomalies": len(flagged),
            "window_hours": hours,
            "threshold": threshold,
            "anomalies": records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/summary", tags=["Metrics"])
def get_metrics_summary(hours: int = Query(24, ge=1, le=72)):
    try:
        conn = db_conn()
        df = pd.read_sql("SELECT * FROM infrastructure_metrics", conn, parse_dates=["timestamp"])
        conn.close()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max() - timedelta(hours=hours)
        window = df[df["timestamp"] >= cutoff]

        return {
            "window_hours": hours,
            "total_records": len(window),
            "avg_cpu_percent": round(window["cpu_percent"].mean(), 2),
            "avg_memory_percent": round(window["memory_percent"].mean(), 2),
            "avg_latency_p95_ms": round(window["latency_p95_ms"].mean(), 2),
            "avg_error_rate": round(window["error_rate"].mean(), 5),
            "max_cpu_percent": round(window["cpu_percent"].max(), 2),
            "max_latency_p95_ms": round(window["latency_p95_ms"].max(), 2),
            "total_container_restarts": int(window["container_restarts"].sum()),
            "ground_truth_anomalies": int(window["is_anomaly"].sum()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/forecast", tags=["Forecasting"])
def get_forecast(steps: int = Query(4, ge=1, le=20)):
    try:
        import pickle
        raw = load_metrics(DB_PATH)
        feat_df = build_forecast_features(raw)
        feature_cols = get_feature_cols()

        with open(f"{MODEL_DIR}/forecaster.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{MODEL_DIR}/forecaster_metadata.json") as f:
            meta = json.load(f)

        recent = feat_df.tail(steps)
        X = recent[feature_cols].values
        preds = model.predict(X)

        return {
            "forecast_target": meta["target"],
            "horizon_minutes": round(steps * 15 / 60, 1),
            "model_mape": meta["metrics"]["mape"],
            "forecasts": [
                {
                    "timestamp": str(row["timestamp"]),
                    "forecast_value": round(float(preds[i]), 2),
                    "breach_probability": round(float(preds[i] > 85), 2),
                }
                for i, (_, row) in enumerate(recent.iterrows())
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze", tags=["Reasoning"])
def analyze(request: AnomalyAnalysisRequest):
    try:
        context = request.model_dump()
        result = analyze_anomaly(context, CHROMA_PATH)
        result.pop("similar_incidents", None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/model/performance", tags=["Models"])
def model_performance():
    try:
        with open(f"{MODEL_DIR}/ensemble_metadata.json") as f:
            ensemble = json.load(f)
        with open(f"{MODEL_DIR}/forecaster_metadata.json") as f:
            forecaster = json.load(f)
        return {
            "anomaly_detection": ensemble["metrics"],
            "ensemble_weights": {"isolation_forest": ensemble["w_iso"], "autoencoder": ensemble["w_ae"]},
            "forecasting": forecaster["metrics"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
