import numpy as np
import pandas as pd
from typing import Optional


COST_CONFIG = {
    "revenue_per_hour": 50000,
    "engineering_cost_per_hour": 150,
    "infrastructure_cost_per_hour": 800,
    "customer_churn_cost_per_incident": 2500,
    "sla_penalty_per_hour": 5000,
}

ANOMALY_IMPACT_PROFILES = {
    "cascade": {
        "revenue_impact_pct": 0.85,
        "affected_users_pct": 0.90,
        "engineering_hours": 2.5,
        "churn_probability": 0.15,
        "sla_breach_probability": 0.80,
    },
    "cpu_spike": {
        "revenue_impact_pct": 0.20,
        "affected_users_pct": 0.25,
        "engineering_hours": 0.75,
        "churn_probability": 0.02,
        "sla_breach_probability": 0.15,
    },
    "memory_leak": {
        "revenue_impact_pct": 0.35,
        "affected_users_pct": 0.40,
        "engineering_hours": 1.5,
        "churn_probability": 0.05,
        "sla_breach_probability": 0.30,
    },
    "latency_surge": {
        "revenue_impact_pct": 0.45,
        "affected_users_pct": 0.55,
        "engineering_hours": 1.0,
        "churn_probability": 0.08,
        "sla_breach_probability": 0.40,
    },
    "error_burst": {
        "revenue_impact_pct": 0.60,
        "affected_users_pct": 0.65,
        "engineering_hours": 1.25,
        "churn_probability": 0.10,
        "sla_breach_probability": 0.55,
    },
    "normal": {
        "revenue_impact_pct": 0.0,
        "affected_users_pct": 0.0,
        "engineering_hours": 0.0,
        "churn_probability": 0.0,
        "sla_breach_probability": 0.0,
    },
}


def estimate_cost_impact(
    anomaly_type: str,
    ensemble_score: float,
    duration_minutes: float = 30.0,
    cost_config: Optional[dict] = None,
) -> dict:
    config = cost_config or COST_CONFIG
    profile = ANOMALY_IMPACT_PROFILES.get(anomaly_type, ANOMALY_IMPACT_PROFILES["normal"])
    severity_multiplier = 0.5 + ensemble_score * 0.5
    duration_hours = duration_minutes / 60

    revenue_loss = (
        config["revenue_per_hour"]
        * profile["revenue_impact_pct"]
        * severity_multiplier
        * duration_hours
    )
    engineering_cost = (
        config["engineering_cost_per_hour"]
        * profile["engineering_hours"]
        * severity_multiplier
    )
    infrastructure_cost = (
        config["infrastructure_cost_per_hour"]
        * severity_multiplier
        * duration_hours
    )
    churn_cost = (
        config["customer_churn_cost_per_incident"]
        * profile["churn_probability"]
        * severity_multiplier
    )
    sla_cost = (
        config["sla_penalty_per_hour"]
        * profile["sla_breach_probability"]
        * severity_multiplier
        * duration_hours
    )

    total = revenue_loss + engineering_cost + infrastructure_cost + churn_cost + sla_cost

    mttr_manual = 47
    mttr_iocc = 1.5
    time_saved_hours = (mttr_manual - mttr_iocc) / 60
    cost_avoided = (
        config["revenue_per_hour"]
        * profile["revenue_impact_pct"]
        * severity_multiplier
        * time_saved_hours
    )

    return {
        "anomaly_type": anomaly_type,
        "ensemble_score": round(ensemble_score, 4),
        "duration_minutes": duration_minutes,
        "severity_multiplier": round(severity_multiplier, 3),
        "revenue_loss_usd": round(revenue_loss, 2),
        "engineering_cost_usd": round(engineering_cost, 2),
        "infrastructure_cost_usd": round(infrastructure_cost, 2),
        "churn_cost_usd": round(churn_cost, 2),
        "sla_penalty_usd": round(sla_cost, 2),
        "total_estimated_cost_usd": round(total, 2),
        "cost_avoided_by_early_detection_usd": round(cost_avoided, 2),
        "affected_users_pct": round(profile["affected_users_pct"] * severity_multiplier * 100, 1),
        "sla_breach_probability": round(profile["sla_breach_probability"] * severity_multiplier, 3),
    }


def estimate_batch(df: pd.DataFrame, duration_minutes: float = 30.0) -> pd.DataFrame:
    results = []
    for _, row in df.iterrows():
        estimate = estimate_cost_impact(
            anomaly_type=str(row.get("anomaly_type", "normal")),
            ensemble_score=float(row.get("ensemble_score", 0.5)),
            duration_minutes=duration_minutes,
        )
        estimate["timestamp"] = row.get("timestamp")
        results.append(estimate)
    return pd.DataFrame(results)


if __name__ == "__main__":
    test_cases = [
        ("cascade", 0.82),
        ("error_burst", 0.65),
        ("latency_surge", 0.55),
        ("cpu_spike", 0.45),
        ("memory_leak", 0.70),
    ]
    print(f"{'Type':<20} {'Score':<8} {'Total Cost':<15} {'Cost Avoided':<15} {'SLA Breach Prob'}")
    print("-" * 75)
    for anomaly_type, score in test_cases:
        result = estimate_cost_impact(anomaly_type, score)
        print(
            f"{anomaly_type:<20} {score:<8} "
            f"${result['total_estimated_cost_usd']:<14,.0f} "
            f"${result['cost_avoided_by_early_detection_usd']:<14,.0f} "
            f"{result['sla_breach_probability']:.1%}"
        )
