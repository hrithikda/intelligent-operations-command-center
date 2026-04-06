import streamlit as st
import pandas as pd
import sqlite3
import json
from datetime import timedelta

from src.ingestion.metrics_generator import generate_infrastructure_metrics, save_to_sqlite
from src.features.feature_engineer import engineer_features, save_features
from src.models.ensemble import predict as ensemble_predict

st.set_page_config(
    page_title="IOCC — Intelligent Operations Command Center",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/iocc.db"
MODEL_DIR = "data/models"


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
    tab1, tab2, tab3 = st.tabs(["Command Center", "Anomaly Explorer", "System Health"])
    with tab1:
        window = command_center(df, hours, threshold)
    with tab2:
        cutoff = df["timestamp"].max() - timedelta(hours=hours)
        window_full = df[df["timestamp"] >= cutoff].copy()
        anomaly_explorer(window_full, threshold)
    with tab3:
        system_health(df)


if __name__ == "__main__":
    main()
