import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("src.api.main.load_dotenv"):
        with patch("src.api.main.load_features"):
            with patch("src.api.main.ensemble_predict"):
                with patch("src.api.main.load_from_sqlite"):
                    with patch("src.api.main.load_metrics"):
                        from src.api.main import app
                        return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "db_available" in data
    assert "models_available" in data


def test_health_returns_version(client):
    response = client.get("/health")
    assert response.json()["version"] == "1.0.0"


def test_docs_available(client):
    response = client.get("/docs")
    assert response.status_code == 200


def test_openapi_schema(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "paths" in schema
    assert "/health" in schema["paths"]
    assert "/api/v1/anomalies" in schema["paths"]
    assert "/api/v1/analyze" in schema["paths"]
