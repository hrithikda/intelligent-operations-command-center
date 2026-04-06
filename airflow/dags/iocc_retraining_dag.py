from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "iocc",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def check_drift(**context):
    import sys
    sys.path.insert(0, "/app")
    import sqlite3
    import numpy as np
    import pandas as pd

    conn = sqlite3.connect("data/iocc.db")
    df = pd.read_sql("SELECT * FROM infrastructure_features", conn, parse_dates=["timestamp"])
    conn.close()

    df = df.sort_values("timestamp")
    split = int(len(df) * 0.8)
    ref = df.iloc[:split]
    curr = df.iloc[split:]

    cols = ["cpu_percent", "memory_percent", "latency_p95_ms", "error_rate"]
    drift_detected = False
    for col in cols:
        ref_mean = ref[col].mean()
        curr_mean = curr[col].mean()
        ref_std = ref[col].std() + 1e-9
        psi = abs(curr_mean - ref_mean) / ref_std
        print(f"PSI for {col}: {psi:.4f}")
        if psi > 0.2:
            drift_detected = True
            print(f"Drift detected in {col}")

    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    return drift_detected


def retrain_anomaly_detector(**context):
    import sys
    sys.path.insert(0, "/app")
    drift = context["ti"].xcom_pull(key="drift_detected", task_ids="check_data_drift")
    if not drift:
        print("No drift detected, skipping retraining.")
        return
    from src.models.anomaly_detector import train
    metrics = train()
    print(f"Anomaly detector retrained: {metrics}")


def retrain_autoencoder(**context):
    import sys
    sys.path.insert(0, "/app")
    drift = context["ti"].xcom_pull(key="drift_detected", task_ids="check_data_drift")
    if not drift:
        print("No drift detected, skipping retraining.")
        return
    from src.models.autoencoder import train
    metrics = train()
    print(f"Autoencoder retrained: {metrics}")


def retrain_ensemble(**context):
    import sys
    sys.path.insert(0, "/app")
    drift = context["ti"].xcom_pull(key="drift_detected", task_ids="check_data_drift")
    if not drift:
        print("No drift detected, skipping retraining.")
        return
    from src.models.ensemble import train
    metrics = train()
    print(f"Ensemble retrained: {metrics}")


def retrain_forecaster(**context):
    import sys
    sys.path.insert(0, "/app")
    from src.models.forecaster import train
    metrics = train()
    print(f"Forecaster retrained: {metrics}")


def evaluate_models(**context):
    import sys, json
    sys.path.insert(0, "/app")
    with open("data/models/ensemble_metadata.json") as f:
        ensemble = json.load(f)
    with open("data/models/forecaster_metadata.json") as f:
        forecaster = json.load(f)

    precision = ensemble["metrics"]["precision"]
    mape = forecaster["metrics"]["mape"]

    print(f"Post-retrain precision: {precision:.4f}")
    print(f"Post-retrain MAPE: {mape:.4f}")

    if precision < 0.80:
        raise ValueError(f"Precision {precision:.4f} below minimum threshold 0.80")
    if mape > 25.0:
        raise ValueError(f"MAPE {mape:.4f} above maximum threshold 25.0")

    print("Model evaluation passed.")


with DAG(
    dag_id="iocc_model_retraining",
    default_args=default_args,
    description="IOCC drift detection and model retraining pipeline",
    schedule_interval="0 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["iocc", "retraining", "mlops"],
) as dag:

    check_drift_task = PythonOperator(
        task_id="check_data_drift",
        python_callable=check_drift,
        provide_context=True,
    )

    retrain_iso = PythonOperator(
        task_id="retrain_isolation_forest",
        python_callable=retrain_anomaly_detector,
        provide_context=True,
    )

    retrain_ae = PythonOperator(
        task_id="retrain_autoencoder",
        python_callable=retrain_autoencoder,
        provide_context=True,
    )

    retrain_ens = PythonOperator(
        task_id="retrain_ensemble",
        python_callable=retrain_ensemble,
        provide_context=True,
    )

    retrain_lgbm = PythonOperator(
        task_id="retrain_forecaster",
        python_callable=retrain_forecaster,
        provide_context=True,
    )

    evaluate = PythonOperator(
        task_id="evaluate_retrained_models",
        python_callable=evaluate_models,
        provide_context=True,
    )

    check_drift_task >> [retrain_iso, retrain_ae, retrain_lgbm]
    retrain_iso >> retrain_ens
    retrain_ae >> retrain_ens
    retrain_ens >> evaluate
    retrain_lgbm >> evaluate
