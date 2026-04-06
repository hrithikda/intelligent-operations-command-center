import streamlit as st
import pandas as pd
import sqlite3
import json
import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.metrics_generator import generate_infrastructure_metrics, save_to_sqlite
from src.features.feature_engineer import engineer_features, save_features
from src.models.ensemble import predict as ensemble_predict
from src.reasoning.rag_engine import analyze_anomaly, build_knowledge_base

st.set_page_config(
    page_title="IOCC — Intelligent Operations Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/iocc.db"
MODEL_DIR = "data/models"
CHROMA_PATH = "data/chroma"


def load_raw() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM infrastructure_metrics", conn, parse_dates=["timestamp"])
    conn.close()
    return df


def load_features_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM infrastructure_features", conn, parse_dates=["timestamp"])
    conn.close()
    return df


def load_ensemble_meta() -> dict:
    with open(f"{MODEL_DIR}/ensemble_metadata.json") as f:
        return json.load(f)


def health_color(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return "#ef4444"
    if value >= warn:
        return "#f59e0b"
    return "#22c55e"


@st.cache_data(ttl=30)
def get_predictions() -> pd.DataFrame:
    features = load_features_df()
    preds = ensemble_predict(features, MODEL_DIR)
    preds = preds.rename(columns={"is_anomaly": "predicted_anomaly"})
    raw = load_raw()[["timestamp", "cpu_percent", "memory_percent", "latency_p95_ms",
                       "error_rate", "container_restarts", "is_anomaly", "anomaly_type"]]
    raw = raw.rename(columns={"is_anomaly": "ground_truth_anomaly"})
    merged = preds.merge(raw, on="timestamp", how="left")
    merged["ground_truth_anomaly"] = merged["ground_truth_anomaly"].fillna(0).astype(int)
    merged["anomaly_type"] = merged["anomaly_type"].fillna("normal")
    return merged


def sidebar():
    st.sidebar.title("⚡ IOCC")
    st.sidebar.caption("Intelligent Operations Command Center")
    st.sidebar.divider()

    meta = load_ensemble_meta()
    m = meta["metrics"]
    st.sidebar.markdown("**Model Performance**")
    st.sidebar.metric("Precision", f"{m['precision']:.1%}")
    st.sidebar.metric("Recall", f"{m['recall']:.1%}")
    st.sidebar.metric("F1 Score", f"{m['f1']:.1%}")
    st.sidebar.metric("False Positive Rate", f"{m['false_positive_rate']:.2%}")
    st.sidebar.divider()

    hours = st.sidebar.slider("Time window (hours)", 1, 72, 24)
    threshold = st.sidebar.slider(
        "Anomaly score threshold",
        min_value=0.05, max_value=0.95,
        value=float(meta["threshold"]),
        step=0.01,
    )
    st.sidebar.divider()
    if st.sidebar.button("Regenerate Data & Retrain", type="secondary"):
        with st.spinner("Regenerating..."):
            df = generate_infrastructure_metrics()
            save_to_sqlite(df)
            feats = engineer_features(df)
            save_features(feats)
            st.cache_data.clear()
            st.success("Data regenerated.")
    return hours, threshold


def command_center(df: pd.DataFrame, hours: int, threshold: float):
    st.markdown("## ⚡ Command Center")
    cutoff = df["timestamp"].max() - timedelta(hours=hours)
    window = df[df["timestamp"] >= cutoff].copy()
    window["flagged"] = (window["ensemble_score"] >= threshold).astype(int)

    total = len(window)
    flagged = int(window["flagged"].sum())
    ground_truth = int(window["ground_truth_anomaly"].sum())
    avg_cpu = window["cpu_percent"].mean()
    avg_mem = window["memory_percent"].mean()
    avg_lat = window["latency_p95_ms"].mean()
    avg_err = window["error_rate"].mean()

    status_color = "#22c55e" if flagged == 0 else "#ef4444"
    status_label = "HEALTHY" if flagged == 0 else f"{flagged} ANOMALIES DETECTED"

    st.markdown(
        f'<div style="background:{status_color}22;border-left:4px solid {status_color};'
        f'padding:12px 20px;border-radius:8px;margin-bottom:1rem">'
        f'<span style="color:{status_color};font-size:1.1rem;font-weight:700">● {status_label}</span>'
        f'<span style="color:#6b7280;font-size:0.85rem;margin-left:1rem">Last {hours}h · {total:,} data points</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Anomalies Flagged", flagged, delta=f"{flagged - ground_truth:+d} vs ground truth", delta_color="inverse")
    c2.metric("Avg CPU %", f"{avg_cpu:.1f}%")
    c3.metric("Avg Memory %", f"{avg_mem:.1f}%")
    c4.metric("Avg Latency p95", f"{avg_lat:.0f}ms")
    c5.metric("Avg Error Rate", f"{avg_err:.3%}")
    c6.metric("Data Points", f"{total:,}")

    st.divider()
    st.markdown("### Real-Time Metrics")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**CPU & Memory %**")
        st.line_chart(window.set_index("timestamp")[["cpu_percent", "memory_percent"]].tail(500), height=200)
        st.markdown("**Latency p95 (ms)**")
        st.line_chart(window.set_index("timestamp")[["latency_p95_ms"]].tail(500), height=200)
    with col_right:
        st.markdown("**Error Rate**")
        st.line_chart(window.set_index("timestamp")[["error_rate"]].tail(500), height=200)
        st.markdown("**Ensemble Anomaly Score**")
        st.line_chart(window.set_index("timestamp")[["ensemble_score"]].tail(500), height=200)
    return window


def anomaly_explorer(window: pd.DataFrame, threshold: float):
    st.markdown("## 🔍 Anomaly Explorer")
    flagged = window[window["ensemble_score"] >= threshold].sort_values("ensemble_score", ascending=False)

    if flagged.empty:
        st.success("No anomalies detected in this time window.")
        return

    st.markdown(f"**{len(flagged)} anomalies detected** — sorted by severity")
    display_cols = ["timestamp", "ensemble_score", "ae_score", "iso_score",
                    "cpu_percent", "memory_percent", "latency_p95_ms",
                    "error_rate", "container_restarts", "anomaly_type"]
    display = flagged[display_cols].copy()
    display["ensemble_score"] = display["ensemble_score"].map("{:.3f}".format)
    display["error_rate"] = display["error_rate"].map("{:.4f}".format)
    display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display.head(50), use_container_width=True)

    st.markdown("### Anomaly Type Breakdown")
    type_counts = flagged["anomaly_type"].value_counts()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(type_counts.rename("count").reset_index())
    with col2:
        st.bar_chart(type_counts)


def llm_reasoning(df: pd.DataFrame, threshold: float):
    st.markdown("## 🤖 LLM Root Cause Analysis")

    if not os.path.exists(f"{CHROMA_PATH}/chroma.sqlite3"):
        st.warning("Knowledge base not built yet.")
        if st.button("Build Knowledge Base"):
            with st.spinner("Building..."):
                build_knowledge_base(CHROMA_PATH)
            st.success("Knowledge base ready.")
            st.rerun()
        return

    flagged = df[df["ensemble_score"] >= threshold].sort_values("ensemble_score", ascending=False)
    if flagged.empty:
        st.info("No anomalies detected to analyze.")
        return

    top = flagged.head(20)
    options = [
        f"{row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} | score={row['ensemble_score']:.3f} | {row['anomaly_type']}"
        for _, row in top.iterrows()
    ]
    selected = st.selectbox("Select an anomaly to analyze:", options)
    idx = options.index(selected)
    row = top.iloc[idx]

    anomaly_context = {
        "timestamp": str(row["timestamp"]),
        "anomaly_type": row["anomaly_type"],
        "ensemble_score": float(row["ensemble_score"]),
        "cpu_percent": float(row["cpu_percent"]),
        "memory_percent": float(row["memory_percent"]),
        "latency_p95_ms": float(row["latency_p95_ms"]),
        "error_rate": float(row["error_rate"]),
        "container_restarts": int(row["container_restarts"]),
    }

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU %", f"{anomaly_context['cpu_percent']:.1f}%")
    col2.metric("Memory %", f"{anomaly_context['memory_percent']:.1f}%")
    col3.metric("Latency p95", f"{anomaly_context['latency_p95_ms']:.0f}ms")
    col4.metric("Error Rate", f"{anomaly_context['error_rate']:.3%}")

    if st.button("Analyze with AI", type="primary"):
        with st.spinner("Retrieving similar incidents and generating root cause analysis..."):
            result = analyze_anomaly(anomaly_context, CHROMA_PATH)

        source_label = "Claude AI" if result.get("source") == "claude" else "Rule-Based Engine"
        impact_colors = {"low": "#22c55e", "medium": "#f59e0b", "high": "#ef4444", "critical": "#7c3aed"}
        impact = result.get("estimated_impact", "medium")
        impact_color = impact_colors.get(impact, "#6b7280")

        st.markdown(
            f'<div style="background:#1e293b;border-radius:12px;padding:20px;margin:1rem 0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">'
            f'<span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">Root Cause Analysis</span>'
            f'<span style="background:#334155;padding:3px 10px;border-radius:8px;font-size:0.75rem;color:#94a3b8">{source_label}</span>'
            f'</div>'
            f'<div style="font-size:1rem;font-weight:600;color:#60a5fa;margin-bottom:8px">'
            f'{result["root_cause_classification"].replace("_", " ").title()}</div>'
            f'<div style="color:#cbd5e1;font-size:0.9rem;margin-bottom:12px">{result["root_cause_explanation"]}</div>'
            f'<div style="display:flex;gap:16px">'
            f'<span style="color:#94a3b8;font-size:0.8rem">Confidence: <strong style="color:#f1f5f9">{result["confidence_score"]:.0%}</strong></span>'
            f'<span style="color:#94a3b8;font-size:0.8rem">Impact: <strong style="color:{impact_color}">{impact.upper()}</strong></span>'
            f'<span style="color:#94a3b8;font-size:0.8rem">Est. Resolution: <strong style="color:#f1f5f9">{result["estimated_resolution_minutes"]} min</strong></span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        col_ev, col_act = st.columns(2)
        with col_ev:
            st.markdown("**Supporting Evidence**")
            for e in result.get("supporting_evidence", []):
                st.markdown(f"- {e}")

        with col_act:
            st.markdown("**Recommended Actions**")
            for i, a in enumerate(result.get("recommended_actions", []), 1):
                st.markdown(f"{i}. {a}")

        similar = result.get("similar_incidents", [])
        if similar:
            st.markdown("**Similar Historical Incidents**")
            for inc in similar:
                with st.expander(f"{inc['title']} (similarity: {inc['similarity_score']:.2f})"):
                    st.markdown(f"**Root Cause:** {inc['root_cause']}")
                    st.markdown(f"**Resolution:** {inc['resolution']}")
                    st.markdown(f"**Impact:** {inc['impact']} | **Duration:** {inc['duration_minutes']} minutes")


def system_health(df: pd.DataFrame):
    st.markdown("## 🩺 System Health")
    latest = df.sort_values("timestamp").iloc[-1]

    metrics = {
        "CPU %": (latest["cpu_percent"], 70, 90),
        "Memory %": (latest["memory_percent"], 75, 90),
        "Latency p95 ms": (latest["latency_p95_ms"], 300, 800),
        "Error Rate %": (latest["error_rate"] * 100, 2, 10),
    }

    cols = st.columns(4)
    for col, (name, (val, warn, crit)) in zip(cols, metrics.items()):
        color = health_color(val, warn, crit)
        col.markdown(
            f'<div style="border:1px solid {color};border-radius:8px;padding:16px;text-align:center">'
            f'<div style="color:{color};font-size:1.6rem;font-weight:700">{val:.1f}</div>'
            f'<div style="color:#6b7280;font-size:0.8rem">{name}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### Model Registry")
    meta = load_ensemble_meta()
    m = meta["metrics"]
    col1, col2, col3 = st.columns(3)
    col1.markdown("**Ensemble Metrics**")
    col1.json({"precision": m["precision"], "recall": m["recall"],
               "f1": m["f1"], "avg_precision": m["average_precision"],
               "fpr": m["false_positive_rate"]})
    col2.markdown("**Ensemble Weights**")
    col2.json({"isolation_forest": meta["w_iso"], "autoencoder": meta["w_ae"]})
    col3.markdown("**Threshold**")
    col3.json({"ensemble_threshold": round(meta["threshold"], 4)})


def main():
    hours, threshold = sidebar()
    df = get_predictions()
    tab1, tab2, tab3, tab4 = st.tabs(["Command Center", "Anomaly Explorer", "LLM Analysis", "System Health"])
    with tab1:
        command_center(df, hours, threshold)
    with tab2:
        cutoff = df["timestamp"].max() - timedelta(hours=hours)
        window_full = df[df["timestamp"] >= cutoff].copy()
        anomaly_explorer(window_full, threshold)
    with tab3:
        cutoff = df["timestamp"].max() - timedelta(hours=hours)
        window_full = df[df["timestamp"] >= cutoff].copy()
        llm_reasoning(window_full, threshold)
    with tab4:
        system_health(df)


if __name__ == "__main__":
    main()
