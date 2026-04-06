from locust import HttpUser, task, between
import json


class IOCCUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(5)
    def get_metrics_summary(self):
        self.client.get("/api/v1/metrics/summary?hours=24")

    @task(4)
    def get_anomalies(self):
        self.client.get("/api/v1/anomalies?hours=24&threshold=0.12&limit=20")

    @task(2)
    def get_forecast(self):
        self.client.get("/api/v1/forecast?steps=4")

    @task(1)
    def get_model_performance(self):
        self.client.get("/api/v1/model/performance")

    @task(2)
    def analyze_anomaly(self):
        self.client.post(
            "/api/v1/analyze",
            json={
                "timestamp": "2026-04-06 14:00:00",
                "anomaly_type": "cascade",
                "ensemble_score": 0.82,
                "cpu_percent": 91.5,
                "memory_percent": 87.2,
                "latency_p95_ms": 1340.0,
                "error_rate": 0.218,
                "container_restarts": 5,
            },
            headers={"Content-Type": "application/json"},
        )
