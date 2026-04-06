from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "iocc",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

def run_metrics_ingestion():
    import sys
    sys.path.insert(0, "/app")
    from src.ingestion.metrics_generator import generate_infrastructure_metrics, save_to_sqlite
    df = generate_infrastructure_metrics(hours=1, anomaly_rate=0.03)
    save_to_sqlite(df)
    print(f"Ingested {len(df)} metrics records")

def run_logs_ingestion():
    import sys
    sys.path.insert(0, "/app")
    from src.ingestion.logs_generator import generate_application_logs, save_to_sqlite
    df = generate_application_logs(hours=1, total_records=700)
    save_to_sqlite(df)
    print(f"Ingested {len(df)} log records")

def run_tickets_ingestion():
    import sys
    sys.path.insert(0, "/app")
    from src.ingestion.tickets_generator import generate_support_tickets, save_to_sqlite
    df = generate_support_tickets(hours=1, base_tickets_per_hour=3.0)
    save_to_sqlite(df)
    print(f"Ingested {len(df)} ticket records")

def run_transactions_ingestion():
    import sys
    sys.path.insert(0, "/app")
    from src.ingestion.transactions_generator import generate_transactions, save_to_sqlite
    df = generate_transactions(hours=1, total_records=300)
    save_to_sqlite(df)
    print(f"Ingested {len(df)} transaction records")

def run_feature_engineering():
    import sys
    sys.path.insert(0, "/app")
    from src.ingestion.metrics_generator import load_from_sqlite
    from src.features.feature_engineer import engineer_features, save_features
    raw = load_from_sqlite()
    features = engineer_features(raw)
    save_features(features)
    print(f"Engineered features shape: {features.shape}")

def validate_data(**context):
    import sys, sqlite3
    sys.path.insert(0, "/app")
    conn = sqlite3.connect("data/iocc.db")
    tables = ["infrastructure_metrics", "application_logs", "support_tickets", "transactions"]
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count} records")
        if count == 0:
            raise ValueError(f"Table {table} is empty after ingestion")
    conn.close()
    print("Data validation passed")

with DAG(
    dag_id="iocc_data_ingestion",
    default_args=default_args,
    description="IOCC multi-source data ingestion pipeline",
    schedule_interval="*/15 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["iocc", "ingestion"],
) as dag:

    ingest_metrics = PythonOperator(
        task_id="ingest_infrastructure_metrics",
        python_callable=run_metrics_ingestion,
    )

    ingest_logs = PythonOperator(
        task_id="ingest_application_logs",
        python_callable=run_logs_ingestion,
    )

    ingest_tickets = PythonOperator(
        task_id="ingest_support_tickets",
        python_callable=run_tickets_ingestion,
    )

    ingest_transactions = PythonOperator(
        task_id="ingest_transactions",
        python_callable=run_transactions_ingestion,
    )

    validate = PythonOperator(
        task_id="validate_ingested_data",
        python_callable=validate_data,
        provide_context=True,
    )

    engineer_feats = PythonOperator(
        task_id="engineer_features",
        python_callable=run_feature_engineering,
    )

    [ingest_metrics, ingest_logs, ingest_tickets, ingest_transactions] >> validate >> engineer_feats
