FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.29.0 \
    streamlit==1.35.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    lightgbm==4.3.0 \
    torch==2.3.0 \
    sentence-transformers==3.0.0 \
    chromadb==0.5.0 \
    sqlalchemy==2.0.30 \
    mlflow==2.13.0 \
    pydantic==2.7.1 \
    pydantic-settings==2.3.0 \
    python-dotenv==1.0.1 \
    httpx==0.27.0 \
    anthropic \
    prometheus-client==0.20.0

COPY . .

EXPOSE 8000 8501

