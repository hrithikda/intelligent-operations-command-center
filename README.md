# Intelligent Operations Command Center

End-to-end operational intelligence platform. Ingests multi-source data, detects anomalies with ensemble ML, forecasts capacity breaches, and generates automated root cause analyses via RAG-powered LLM reasoning.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-9ACD32?style=flat)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-8B5CF6?style=flat)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=flat&logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)

---

## Results

| Metric | Value |
|--------|-------|
| Anomaly Detection Precision | 94.4% |
| Anomaly Detection Recall | 77.1% |
| F1 Score | 84.9% |
| False Positive Rate | 0.15% |
| Area Under Precision-Recall Curve | 0.8087 |
| Forecasting MAPE | 14.84% |
| Improvement over Naive Baseline | 35.7% |
| Features Engineered | 133 |
| Inference Throughput | 17,280+ events per run |

---

## Architecture

The platform is organized into five sequential layers.

**Ingestion** — Python generators simulate four enterprise data sources: infrastructure metrics sampled at 15-second intervals, application logs with realistic request distributions, support tickets with priority and resolution tracking, and transaction records with fraud pattern injection. Each generator produces configurable anomaly patterns at a 3% base rate. Apache Airflow DAGs handle orchestration, schema validation, and dead-letter queue management.

**Feature Engineering** — 133 features computed across six groups: temporal cyclical encoding (hour and day-of-week sine/cosine), rolling statistics over four windows (1min, 3min, 15min, 1hr), per-metric z-scores against rolling baselines, lag features at 1/4/15/60-minute offsets, rate-of-change deltas and percentage changes, and cross-metric correlation features including rolling error-latency correlation and composite anomaly scores.

**Anomaly Detection** — Two models run in ensemble. Isolation Forest uses 200 estimators with 80% feature fraction and bootstrapping to score multivariate point anomalies across all 133 features. A PyTorch autoencoder (encoder: input→64→32→16, decoder: 16→32→64→input, BatchNorm + Dropout) is trained exclusively on normal operational patterns — reconstruction error on anomalous inputs is used as the anomaly signal. Ensemble weights are learned via F1 maximization on a held-out validation set.

**Forecasting** — LightGBM regressor with 500 estimators and early stopping trained on 9 lag features, 4 rolling window statistics per window, and temporal cyclical encodings. Prophet provides the naive seasonal baseline. Model achieves 14.84% MAPE at a 1-minute forecast horizon, a 35.7% improvement over the baseline. MLflow tracks all experiments with parameters, metrics, and model artifacts.

**Reasoning** — Detected anomalies are embedded using Sentence Transformers (all-MiniLM-L6-v2) and matched against a ChromaDB vector store of historical incidents by cosine similarity. The top-3 retrieved incidents plus structured anomaly context are passed to a prompt chain targeting the Claude API, which returns a JSON object containing root cause classification, confidence score, supporting evidence, recommended remediation steps, estimated impact severity, and estimated resolution time. A deterministic rule-based engine produces an identical output schema when the API is unavailable.

---

## Stack

| Layer | Technology |
|-------|------------|
| Data Ingestion | Python, Apache Airflow |
| Storage | SQLite (development), PostgreSQL + TimescaleDB (production) |
| Feature Engineering | Pandas, NumPy |
| Anomaly Detection | Scikit-learn (Isolation Forest), PyTorch (Autoencoder) |
| Forecasting | LightGBM |
| Vector Store | ChromaDB, Sentence Transformers |
| LLM Reasoning | Anthropic Claude API, rule-based fallback |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Deployment | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Dashboard

Four views accessible via the Streamlit interface at `:8501`.

**Command Center** — Live system status banner, six KPI metrics (anomalies flagged, CPU, memory, latency p95, error rate, data points), real-time line charts for all key signals including ensemble anomaly score with timestamp tooltips.

**Anomaly Explorer** — All detected anomalies sorted by ensemble score with full metric breakdown per row: individual ISO and AE scores, CPU, memory, latency, error rate, container restarts, and injected anomaly type. Anomaly type breakdown table and bar chart.

**LLM Analysis** — Select any detected anomaly from a ranked dropdown. Triggers RAG retrieval against the ChromaDB incident knowledge base and LLM prompt chain. Returns root cause classification, confidence score, estimated impact severity, estimated resolution time, supporting evidence, recommended remediation actions, and expandable similar historical incident cards with similarity scores.

**System Health** — Color-coded health indicators with warn and critical thresholds per metric. Model registry panel showing ensemble precision/recall/F1, ensemble weights, operating threshold, and forecaster MAPE.

---

## Repository Structure

intelligent-operations-command-center/
├── src/
│   ├── ingestion/          # Data generators for all four sources
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Isolation Forest, Autoencoder, Ensemble, LightGBM
│   ├── reasoning/          # RAG engine, ChromaDB, LLM prompt chain
│   ├── api/                # FastAPI application
│   ├── dashboard/          # Streamlit command center
│   └── evaluation/         # Evaluation framework
├── airflow/dags/           # Orchestration DAGs
├── docker/                 # Dockerfile
├── docs/                   # DECISIONS.md
├── tests/                  # Unit and integration tests
├── docker-compose.yml
├── run_dashboard.py
└── requirements.txt

---

## Quick Start

### Local
```bash
git clone https://github.com/hrithikda/intelligent-operations-command-center.git
cd intelligent-operations-command-center
pip install -r requirements.txt
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

python -m src.ingestion.metrics_generator
python -m src.ingestion.logs_generator
python -m src.ingestion.tickets_generator
python -m src.ingestion.transactions_generator
python -m src.features.feature_engineer
python -m src.models.anomaly_detector
python -m src.models.autoencoder
python -m src.models.ensemble
python -m src.models.forecaster
python -m src.reasoning.rag_engine

streamlit run run_dashboard.py    # dashboard on :8501
python -m src.api.main            # API on :8000
```

### Docker
```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System health and model availability |
| GET | /api/v1/anomalies | Detected anomalies with configurable threshold and time window |
| GET | /api/v1/metrics/summary | Aggregated operational KPIs |
| GET | /api/v1/forecast | LightGBM CPU load forecast |
| POST | /api/v1/analyze | RAG and LLM root cause analysis |
| GET | /api/v1/model/performance | Ensemble and forecaster metrics |

Interactive documentation available at `/docs`.

---

## Evaluation

| Layer | Metrics |
|-------|---------|
| Anomaly Detection | Precision, Recall, F1, AUPRC, FPR at operating threshold |
| Forecasting | MAE, RMSE, MAPE, naive baseline comparison |
| LLM Output | Root cause classification accuracy, actionability, retrieval relevance |
| System | API response time p95, end-to-end ingestion-to-alert latency |

---

## Architectural Decisions

All major technical decisions — model selection, storage choices, framework tradeoffs — are documented with rationale and alternatives considered in [docs/DECISIONS.md](docs/DECISIONS.md).

---

## Author

Hrithik Dasharatha Angadi
MS Information Systems, University of Illinois Chicago
hrithikda@gmail.com · [GitHub](https://github.com/hrithikda) · [LinkedIn](https://linkedin.com/in/hrithikda)