import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.models.anomaly_detector import FEATURE_COLS, train_test_split_temporal


FORECAST_TARGET = "cpu_percent"
FORECAST_HORIZON = 4  # 4 steps = 1 minute ahead at 15s intervals
LAG_FEATURES = [1, 2, 4, 8, 12, 24, 48, 96, 240]
ROLLING_WINDOWS = [4, 12, 60, 240]


def load_metrics(db_path: str = "data/iocc.db") -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM infrastructure_metrics", conn, parse_dates=["timestamp"])
    conn.close()
    return df.sort_values("timestamp").reset_index(drop=True)


def build_forecast_features(df: pd.DataFrame, target: str = FORECAST_TARGET) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"])

    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    df["is_business_hours"] = ((ts.dt.hour >= 9) & (ts.dt.hour < 18) & (ts.dt.dayofweek < 5)).astype(int)

    for lag in LAG_FEATURES:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"{target}_roll_mean_{w}"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std_{w}"] = df[target].shift(1).rolling(w, min_periods=1).std().fillna(0)
        df[f"{target}_roll_max_{w}"] = df[target].shift(1).rolling(w, min_periods=1).max()

    df[f"{target}_diff_1"] = df[target].diff(1)
    df[f"{target}_diff_4"] = df[target].diff(4)
    df["target"] = df[target].shift(-FORECAST_HORIZON)

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_cols(target: str = FORECAST_TARGET) -> list:
    cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_business_hours"]
    cols += [f"{target}_lag_{lag}" for lag in LAG_FEATURES]
    for w in ROLLING_WINDOWS:
        cols += [f"{target}_roll_mean_{w}", f"{target}_roll_std_{w}", f"{target}_roll_max_{w}"]
    cols += [f"{target}_diff_1", f"{target}_diff_4"]
    return cols


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def train(db_path: str = "data/iocc.db", model_dir: str = "data/models") -> dict:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("iocc_forecasting")

    print("Loading and preparing data...")
    raw = load_metrics(db_path)
    df = build_forecast_features(raw)

    feature_cols = get_feature_cols()
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    print(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lightgbm_forecaster"):
        mlflow.log_params(params)
        mlflow.log_param("forecast_horizon_steps", FORECAST_HORIZON)
        mlflow.log_param("forecast_target", FORECAST_TARGET)

        print("Training LightGBM forecaster...")
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape_score = mape(y_test, y_pred)
        naive_pred = test_df[f"{FORECAST_TARGET}_lag_1"].values[:len(y_test)]
        naive_mape = mape(y_test, naive_pred)
        mape_improvement = (naive_mape - mape_score) / naive_mape * 100

        metrics = {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mape": round(mape_score, 4),
            "naive_mape": round(naive_mape, 4),
            "mape_improvement_pct": round(mape_improvement, 2),
            "best_iteration": int(model.best_iteration_),
            "forecast_horizon_minutes": round(FORECAST_HORIZON * 15 / 60, 1),
        }

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.lightgbm.log_model(model, "lightgbm_model")

        print("\nLightGBM Forecaster Results:")
        print(f"  MAE:                  {mae:.4f}%")
        print(f"  RMSE:                 {rmse:.4f}%")
        print(f"  MAPE:                 {mape_score:.2f}%")
        print(f"  Naive baseline MAPE:  {naive_mape:.2f}%")
        print(f"  MAPE improvement:     {mape_improvement:.1f}% over naive")
        print(f"  Forecast horizon:     {FORECAST_HORIZON * 15 / 60:.0f} minutes ahead")
        print(f"  Best iteration:       {model.best_iteration_}")

    with open(f"{model_dir}/forecaster.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{model_dir}/forecaster_metadata.json", "w") as f:
        json.dump({"metrics": metrics, "feature_cols": feature_cols,
                   "target": FORECAST_TARGET, "horizon_steps": FORECAST_HORIZON}, f, indent=2)

    print(f"\nForecaster saved to {model_dir}/")
    return metrics


def predict(df: pd.DataFrame, model_dir: str = "data/models") -> pd.DataFrame:
    with open(f"{model_dir}/forecaster.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/forecaster_metadata.json") as f:
        meta = json.load(f)

    feat_df = build_forecast_features(df)
    feature_cols = meta["feature_cols"]
    X = feat_df[feature_cols].values
    preds = model.predict(X)

    result = feat_df[["timestamp"]].copy()
    result["forecast_cpu"] = preds.round(2)
    result["actual_cpu"] = feat_df["target"].values
    result["breach_probability"] = (preds > 85).astype(float)
    return result


if __name__ == "__main__":
    metrics = train()
    print("\nForecasting training complete.")
