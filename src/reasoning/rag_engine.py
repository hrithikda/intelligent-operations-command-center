import json
import os
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "incident_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"

INCIDENT_TEMPLATES = [
    {"type": "cascade", "title": "Full cascade failure across services",
     "description": "CPU spiked to 95%, memory at 88%, latency exceeded 1200ms, error rate hit 22%, container restarts surged to 7. All metrics degraded simultaneously indicating a cascade failure.",
     "root_cause": "Upstream dependency failure caused cascading timeouts across all services.",
     "resolution": "Identified failing upstream service, implemented circuit breaker, rolled back recent deployment, restored normal operation in 23 minutes.",
     "impact": "High", "duration_minutes": 23},
    {"type": "cpu_spike", "title": "Sustained CPU saturation on compute nodes",
     "description": "CPU utilization climbed to 98% over 8 minutes. Memory stable. Latency increased 40%. No error rate change.",
     "root_cause": "Runaway background job consuming all available CPU cycles without throttling.",
     "resolution": "Identified and killed runaway process, added CPU limits to job scheduler, incident resolved in 11 minutes.",
     "impact": "Medium", "duration_minutes": 11},
    {"type": "memory_leak", "title": "Progressive memory exhaustion in auth service",
     "description": "Memory usage grew linearly from 60% to 94% over 45 minutes. CPU normal. Latency gradually increased. No errors until OOM.",
     "root_cause": "Memory leak in session cache — objects not being garbage collected after session expiry.",
     "resolution": "Restarted auth service instances, deployed hotfix for session cache cleanup, memory stabilized at 55%.",
     "impact": "High", "duration_minutes": 52},
    {"type": "latency_surge", "title": "Database query latency spike causing API timeout",
     "description": "API latency p95 jumped from 120ms to 1800ms. Error rate increased to 8%. CPU and memory normal.",
     "root_cause": "Missing index on high-traffic query after schema migration. Full table scans causing latency explosion.",
     "resolution": "Added missing index, query latency dropped to 95ms within 2 minutes of deployment.",
     "impact": "High", "duration_minutes": 18},
    {"type": "error_burst", "title": "500 error storm from payment service",
     "description": "Error rate spiked from 1% to 35% in 3 minutes. Latency doubled. CPU slightly elevated. Errors concentrated in payment-service.",
     "root_cause": "Third-party payment gateway returning 503s due to their maintenance window not communicated.",
     "resolution": "Enabled fallback payment processor, errors dropped to baseline. Notified vendor of SLA breach.",
     "impact": "Critical", "duration_minutes": 31},
    {"type": "cascade", "title": "Network partition causing split-brain",
     "description": "Intermittent errors across all services. Latency highly variable. Some requests succeeding, others failing. Container restarts elevated.",
     "root_cause": "Network switch failure causing partial partition. Services unable to reach consensus on state.",
     "resolution": "Failed over to secondary network path, services recovered. Root cause hardware replaced.",
     "impact": "Critical", "duration_minutes": 45},
    {"type": "cpu_spike", "title": "Crypto mining malware on compromised instance",
     "description": "CPU at 100% on three instances simultaneously. Normal traffic patterns. No latency or error impact initially.",
     "root_cause": "Compromised container image running cryptomining workload after supply chain attack.",
     "resolution": "Isolated instances, rotated all credentials, rebuilt from clean images, added image scanning to CI/CD.",
     "impact": "Critical", "duration_minutes": 67},
    {"type": "latency_surge", "title": "CDN cache invalidation storm",
     "description": "Origin server latency increased 10x. CDN hit rate dropped from 94% to 12%. Error rate minimal but throughput collapsed.",
     "root_cause": "Bulk cache invalidation triggered by content deployment flushed entire CDN cache simultaneously.",
     "resolution": "Implemented staged cache warming, restored CDN hit rate to 91% within 20 minutes.",
     "impact": "Medium", "duration_minutes": 25},
    {"type": "memory_leak", "title": "Connection pool exhaustion in API gateway",
     "description": "Memory grew steadily. Active connections metric climbed. Eventually new requests queued indefinitely.",
     "root_cause": "Connection pool not releasing connections on timeout. Leaked connections accumulating over hours.",
     "resolution": "Deployed connection pool fix, restarted gateway with connection drain, added connection leak alerting.",
     "impact": "High", "duration_minutes": 38},
    {"type": "error_burst", "title": "Config change caused authentication failures",
     "description": "401 errors spiked to 60% of requests immediately after deployment. CPU and memory unaffected.",
     "root_cause": "Misconfigured JWT secret in new deployment. Token validation failing for all requests.",
     "resolution": "Immediate rollback of deployment, authentication restored within 4 minutes.",
     "impact": "Critical", "duration_minutes": 9},
    {"type": "cascade", "title": "Disk I/O saturation causing write failures",
     "description": "Disk I/O maxed at 200 MB/s. Write latency exceeded 30 seconds. Services queuing indefinitely. Memory growing as buffers filled.",
     "root_cause": "Log rotation misconfiguration caused simultaneous rotation of 200GB across all services.",
     "resolution": "Stopped log rotation, cleared disk space, staggered future rotations across time windows.",
     "impact": "High", "duration_minutes": 29},
    {"type": "latency_surge", "title": "Thundering herd after cache server restart",
     "description": "All services simultaneously hitting database after cache restart. Database CPU at 100%. API latency at 5000ms.",
     "root_cause": "Cache warmup not implemented. All services fell through to database simultaneously after cache restart.",
     "resolution": "Implemented probabilistic cache expiry and warmup strategy. Database load normalized in 15 minutes.",
     "impact": "High", "duration_minutes": 22},
]


def build_knowledge_base(chroma_path: str = CHROMA_PATH) -> None:
    Path(chroma_path).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    encoder = SentenceTransformer(EMBED_MODEL)
    texts = [
        f"{inc['title']}. {inc['description']} Root cause: {inc['root_cause']} Resolution: {inc['resolution']}"
        for inc in INCIDENT_TEMPLATES
    ]
    embeddings = encoder.encode(texts, show_progress_bar=False).tolist()

    collection.add(
        ids=[f"incident_{i}" for i in range(len(INCIDENT_TEMPLATES))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "type": inc["type"],
            "title": inc["title"],
            "root_cause": inc["root_cause"],
            "resolution": inc["resolution"],
            "impact": inc["impact"],
            "duration_minutes": inc["duration_minutes"],
        } for inc in INCIDENT_TEMPLATES],
    )
    print(f"Knowledge base built with {len(INCIDENT_TEMPLATES)} incidents.")


def retrieve_similar_incidents(
    anomaly_context: dict,
    chroma_path: str = CHROMA_PATH,
    n_results: int = 3,
) -> list:
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(COLLECTION_NAME)
    encoder = SentenceTransformer(EMBED_MODEL)

    query_text = (
        f"Anomaly type {anomaly_context.get('anomaly_type', 'unknown')}. "
        f"CPU {anomaly_context.get('cpu_percent', 0):.1f}%, "
        f"Memory {anomaly_context.get('memory_percent', 0):.1f}%, "
        f"Latency {anomaly_context.get('latency_p95_ms', 0):.0f}ms, "
        f"Error rate {anomaly_context.get('error_rate', 0):.3f}, "
        f"Anomaly score {anomaly_context.get('ensemble_score', 0):.3f}."
    )
    query_embedding = encoder.encode([query_text], show_progress_bar=False).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    incidents = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        incidents.append({
            "title": meta["title"],
            "root_cause": meta["root_cause"],
            "resolution": meta["resolution"],
            "impact": meta["impact"],
            "duration_minutes": meta["duration_minutes"],
            "similarity_score": round(1 - results["distances"][0][i], 3),
        })
    return incidents


def build_prompt(anomaly_context: dict, similar_incidents: list) -> str:
    incidents_text = ""
    for i, inc in enumerate(similar_incidents, 1):
        incidents_text += (
            f"\nIncident {i} (similarity: {inc['similarity_score']}):\n"
            f"  Title: {inc['title']}\n"
            f"  Root Cause: {inc['root_cause']}\n"
            f"  Resolution: {inc['resolution']}\n"
            f"  Impact: {inc['impact']} | Duration: {inc['duration_minutes']} minutes\n"
        )

    return f"""You are an expert Site Reliability Engineer analyzing an operational anomaly.

DETECTED ANOMALY:
- Timestamp: {anomaly_context.get('timestamp', 'unknown')}
- Anomaly Type: {anomaly_context.get('anomaly_type', 'unknown')}
- Ensemble Score: {anomaly_context.get('ensemble_score', 0):.3f}
- CPU: {anomaly_context.get('cpu_percent', 0):.1f}%
- Memory: {anomaly_context.get('memory_percent', 0):.1f}%
- Latency p95: {anomaly_context.get('latency_p95_ms', 0):.0f}ms
- Error Rate: {anomaly_context.get('error_rate', 0):.3%}
- Container Restarts: {anomaly_context.get('container_restarts', 0)}

SIMILAR HISTORICAL INCIDENTS:
{incidents_text}

Based on the anomaly data and similar historical incidents, provide a structured analysis.
Respond ONLY with valid JSON in exactly this format:
{{
  "root_cause_classification": "one of: cascade_failure, cpu_saturation, memory_exhaustion, latency_degradation, error_storm, network_issue, unknown",
  "confidence_score": 0.0,
  "root_cause_explanation": "2-3 sentence explanation of likely root cause",
  "supporting_evidence": ["evidence point 1", "evidence point 2", "evidence point 3"],
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "estimated_impact": "one of: low, medium, high, critical",
  "estimated_resolution_minutes": 0,
  "similar_incident_reference": "title of most relevant historical incident"
}}"""


def analyze_anomaly(
    anomaly_context: dict,
    chroma_path: str = CHROMA_PATH,
    anthropic_api_key: Optional[str] = None,
) -> dict:
    similar_incidents = retrieve_similar_incidents(anomaly_context, chroma_path)
    prompt = build_prompt(anomaly_context, similar_incidents)

    api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    if api_key and api_key != "your_key_here":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()
            result = json.loads(raw)
            result["source"] = "claude"
            result["similar_incidents"] = similar_incidents
            return result
        except Exception as e:
            print(f"Claude API error: {e}, falling back to rule-based.")

    return rule_based_analysis(anomaly_context, similar_incidents)


def rule_based_analysis(anomaly_context: dict, similar_incidents: list) -> dict:
    cpu = anomaly_context.get("cpu_percent", 0)
    memory = anomaly_context.get("memory_percent", 0)
    latency = anomaly_context.get("latency_p95_ms", 0)
    error_rate = anomaly_context.get("error_rate", 0)
    restarts = anomaly_context.get("container_restarts", 0)
    anomaly_type = anomaly_context.get("anomaly_type", "normal")

    if anomaly_type == "cascade" or (cpu > 85 and memory > 80 and latency > 800):
        classification = "cascade_failure"
        explanation = f"Multiple metrics degraded simultaneously: CPU {cpu:.1f}%, memory {memory:.1f}%, latency {latency:.0f}ms. Pattern consistent with cascade failure triggered by upstream dependency."
        actions = ["Check upstream service dependencies", "Enable circuit breakers", "Review recent deployments", "Scale compute resources"]
        impact = "critical"
        confidence = 0.85
    elif anomaly_type == "cpu_spike" or cpu > 90:
        classification = "cpu_saturation"
        explanation = f"CPU at {cpu:.1f}% with memory stable at {memory:.1f}%. Isolated CPU saturation suggests runaway process or compute-intensive workload."
        actions = ["Identify top CPU consuming processes", "Check for runaway jobs", "Review scheduled tasks", "Consider horizontal scaling"]
        impact = "high"
        confidence = 0.80
    elif anomaly_type == "memory_leak" or memory > 90:
        classification = "memory_exhaustion"
        explanation = f"Memory at {memory:.1f}% with gradual growth pattern. Consistent with memory leak in long-running service."
        actions = ["Check heap dumps for memory leaks", "Review object lifecycle management", "Restart affected services", "Deploy memory leak fix"]
        impact = "high"
        confidence = 0.78
    elif anomaly_type == "latency_surge" or latency > 1000:
        classification = "latency_degradation"
        explanation = f"Latency p95 at {latency:.0f}ms with error rate {error_rate:.2%}. Consistent with database bottleneck or downstream service degradation."
        actions = ["Check database query performance", "Review downstream service health", "Check for missing indexes", "Enable request caching"]
        impact = "high"
        confidence = 0.75
    elif anomaly_type == "error_burst" or error_rate > 0.1:
        classification = "error_storm"
        explanation = f"Error rate at {error_rate:.2%} with latency {latency:.0f}ms. Pattern consistent with configuration error or downstream dependency failure."
        actions = ["Check recent deployments", "Review downstream dependencies", "Verify configuration changes", "Enable fallback mechanisms"]
        impact = "critical"
        confidence = 0.82
    else:
        classification = "unknown"
        explanation = "Anomaly detected but pattern does not match known failure modes. Manual investigation recommended."
        actions = ["Review recent changes", "Check all system logs", "Escalate to on-call engineer"]
        impact = "medium"
        confidence = 0.50

    ref = similar_incidents[0]["title"] if similar_incidents else "No similar incident found"
    return {
        "root_cause_classification": classification,
        "confidence_score": confidence,
        "root_cause_explanation": explanation,
        "supporting_evidence": [
            f"CPU: {cpu:.1f}%", f"Memory: {memory:.1f}%",
            f"Latency p95: {latency:.0f}ms", f"Error rate: {error_rate:.3%}",
        ],
        "recommended_actions": actions,
        "estimated_impact": impact,
        "estimated_resolution_minutes": similar_incidents[0]["duration_minutes"] if similar_incidents else 30,
        "similar_incident_reference": ref,
        "source": "rule_based",
        "similar_incidents": similar_incidents,
    }


if __name__ == "__main__":
    print("Building incident knowledge base...")
    build_knowledge_base()

    test_anomaly = {
        "timestamp": "2026-04-06 14:00:00",
        "anomaly_type": "cascade",
        "ensemble_score": 0.82,
        "cpu_percent": 91.5,
        "memory_percent": 87.2,
        "latency_p95_ms": 1340.0,
        "error_rate": 0.218,
        "container_restarts": 5,
    }

    print("\nAnalyzing test anomaly...")
    result = analyze_anomaly(test_anomaly)
    print(f"\nRoot Cause: {result['root_cause_classification']}")
    print(f"Confidence: {result['confidence_score']:.0%}")
    print(f"Explanation: {result['root_cause_explanation']}")
    print(f"Actions: {result['recommended_actions']}")
    print(f"Impact: {result['estimated_impact']}")
    print(f"Source: {result['source']}")
