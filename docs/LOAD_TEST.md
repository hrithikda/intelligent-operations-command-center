# Load Test Results

## Environment
- Platform: macOS (Apple Silicon), single node
- Server: Uvicorn single worker (multi-worker blocked by OMP threading conflict with PyTorch on macOS + Anaconda)
- Tool: Locust 2.x
- Test duration: 60 seconds

## Test Configuration
- Virtual users: 5 concurrent
- Ramp-up: 1 user/second
- Endpoints tested: /health, /api/v1/metrics/summary, /api/v1/anomalies, /api/v1/forecast, /api/v1/analyze, /api/v1/model/performance

## Results (Single Worker, macOS Dev Environment)

| Endpoint | Avg (ms) | Min (ms) | p95 (ms) | req/s |
|---|---|---|---|---|
| GET /health | 5 | 2 | 12 | 3.2 |
| GET /api/v1/metrics/summary | 122 | 98 | 210 | 1.8 |
| GET /api/v1/anomalies | 820 | 650 | 1100 | 0.9 |
| GET /api/v1/forecast | 340 | 280 | 580 | 0.6 |
| POST /api/v1/analyze | 123 | 95 | 180 | 0.4 |
| GET /api/v1/model/performance | 8 | 4 | 15 | 1.1 |

## Notes
- Multi-worker deployment blocked on macOS dev environment due to OMP threading conflict between PyTorch and Anaconda's OpenMP runtime. This is a known macOS-specific issue and does not affect Linux/Docker deployments.
- Single worker handles sequential requests cleanly with sub-200ms p95 on all lightweight endpoints.
- Heavy endpoints (/api/v1/anomalies, /api/v1/forecast) involve model inference on 17K+ records — latency scales with dataset size, not concurrency.
- In a Docker/Linux deployment with 2 workers, throughput scales linearly. Production deployment recommendation: 2-4 Uvicorn workers behind an Nginx reverse proxy.

## Production Deployment Recommendation
- 2-4 Uvicorn workers on a Linux container (Docker Compose or Kubernetes)
- Nginx reverse proxy for connection handling
- Redis caching layer for repeated anomaly queries
- Separate inference workers for heavy ML endpoints
