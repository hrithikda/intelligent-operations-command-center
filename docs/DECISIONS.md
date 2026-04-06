# DECISIONS.md — Architectural Decision Record

## 1. LightGBM over XGBoost for Forecasting
LightGBM trains significantly faster on tabular time series data and uses less memory. For a portfolio project running on a single node, training speed matters for the retraining loop. XGBoost would produce similar accuracy but at higher compute cost. LightGBM's leaf-wise tree growth also handles the irregular temporal patterns in operational data better than XGBoost's level-wise approach.

## 2. ChromaDB over Pinecone or Weaviate
ChromaDB runs locally with zero infrastructure overhead — no API keys, no managed service cost, no network dependency. For a portfolio project, this is the right tradeoff. Pinecone would add production scalability but requires a managed account and introduces a hard dependency on an external service. ChromaDB's Python-native API also makes the RAG pipeline simpler to understand and explain in interviews.

## 3. Autoencoder over One-Class SVM for Anomaly Detection
The autoencoder learns complex non-linear normal patterns that One-Class SVM cannot capture without extensive kernel engineering. PyTorch autoencoders also scale better with feature dimensionality — at 133 features, kernel methods become computationally expensive. The autoencoder's reconstruction error provides a more interpretable anomaly score than SVM's decision boundary distance.

## 4. Ensemble over Single Model
Neither Isolation Forest nor the autoencoder alone achieves target precision. Isolation Forest catches point anomalies well but struggles with contextual anomalies. The autoencoder excels at contextual anomalies but misses sudden point deviations. The ensemble with learned weights combines both strengths. Weight search on a validation set found the autoencoder dominates (weight 1.0) on this data, but the ensemble infrastructure allows rebalancing as new data sources are added.

## 5. Streamlit over React for Dashboard
Target roles are Data Scientist, ML Engineer, Data Engineer, and AI/GenAI Engineer — not frontend engineer. Streamlit produces a portfolio-ready interface in a fraction of the time React would require, and is universally accepted at every target role level. The FastAPI backend provides the production engineering signal; Streamlit is the communication layer on top.

## 6. FastAPI over Flask
FastAPI provides automatic OpenAPI/Swagger documentation, native Pydantic validation, async support, and significantly better performance than Flask. The auto-generated docs at /docs demonstrate API design quality to interviewers without additional documentation effort. FastAPI's type hint integration also produces cleaner, more maintainable code.

## 7. SQLite over PostgreSQL for Phase 1
SQLite requires zero infrastructure setup, enabling faster iteration during Phase 1. The storage layer is fully abstracted behind load/save functions, making the migration to PostgreSQL in Phase 2 a configuration change rather than a code rewrite. For a single-node portfolio deployment, SQLite handles 17K+ records with sub-second query times.

## 8. Sentence Transformers over OpenAI Embeddings
Sentence Transformers run locally with no API cost and no network dependency. all-MiniLM-L6-v2 produces high-quality embeddings for operational text at 80x lower cost than OpenAI's embedding API. For a RAG pipeline over a small knowledge base of 12 incidents, local embeddings are the correct engineering choice.

## 9. Rule-Based Fallback for LLM Reasoning
The LLM reasoning engine falls back to rule-based analysis when the Claude API is unavailable or has no credits. This makes the system resilient and demonstrable without API dependency. The fallback produces the same structured output format, so the dashboard renders identically regardless of which backend generated the analysis.

## 10. Docker Compose over Kubernetes
Kubernetes introduces significant operational complexity that is not justified for a portfolio project. Docker Compose achieves the core goal — single-command reproducible deployment — with a fraction of the configuration overhead. The architecture is designed to be Kubernetes-compatible (stateless services, environment-based config, health check endpoints) so migration is straightforward if needed.
