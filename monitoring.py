import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict
import re
import os

# --- Log parsing function ---
def parse_logs(filepath="app.log"):
    if not os.path.exists(filepath):
        st.warning(f"Log file '{filepath}' not found. Please check the path or generate logs first.")
        return [], [], [], [], [], [], []

    daily_metrics = defaultdict(lambda: {
        "predictions": 0,
        "errors": 0,
        "f1": [],
        "accuracy": [],
        "roc_auc": [],
        "drift": []
    })

    timestamp_pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"

    with open(filepath, "r") as f:
        for line in f:
            ts_match = re.search(timestamp_pattern, line)
            if not ts_match:
                continue
            ts = datetime.strptime(ts_match.group(), "%Y-%m-%d %H:%M:%S").date()

            if "Prediction made" in line:
                daily_metrics[ts]["predictions"] += 1
            if "ERROR" in line:
                daily_metrics[ts]["errors"] += 1
            if "METRICS" in line:
                try:
                    metrics_match = re.search(
                        r"f1: ([0-9.]+) \| accuracy: ([0-9.]+) \| roc_auc: ([0-9.]+) \| drift_score: ([0-9.]+)", line)
                    if metrics_match:
                        f1, acc, auc, drift = map(float, metrics_match.groups())
                        daily_metrics[ts]["f1"].append(f1)
                        daily_metrics[ts]["accuracy"].append(acc)
                        daily_metrics[ts]["roc_auc"].append(auc)
                        daily_metrics[ts]["drift"].append(drift)
                except:
                    continue

    dates = sorted(daily_metrics.keys())
    predictions = [daily_metrics[d]["predictions"] for d in dates]
    errors = [daily_metrics[d]["errors"] for d in dates]
    f1_scores = [np.mean(daily_metrics[d]["f1"]) if daily_metrics[d]["f1"] else np.nan for d in dates]
    accuracy_scores = [np.mean(daily_metrics[d]["accuracy"]) if daily_metrics[d]["accuracy"] else np.nan for d in dates]
    roc_auc_scores = [np.mean(daily_metrics[d]["roc_auc"]) if daily_metrics[d]["roc_auc"] else np.nan for d in dates]
    drift_scores = [np.mean(daily_metrics[d]["drift"]) if daily_metrics[d]["drift"] else np.nan for d in dates]

    return dates, predictions, errors, f1_scores, accuracy_scores, roc_auc_scores, drift_scores

# --- Load data from logs and create metrics ---
def load_data_from_logs(filepath="app.log"):
    dates, predictions, errors, f1_scores, accuracy_scores, roc_auc_scores, drift_scores = parse_logs(filepath)
    if not dates:
        return None

    return {
        'dates': pd.to_datetime(dates),
        'daily_volume': predictions,
        'errors': errors,
        'f1': f1_scores,
        'accuracy': accuracy_scores,
        'roc_auc': roc_auc_scores,
        'drift_scores': drift_scores
    }

# --- Streamlit app ---
st.set_page_config(page_title="Attrition Predictor Monitoring", layout="wide")
st.title("Attrition Predictor Model Monitoring Dashboard")

# Sidebar controls
st.sidebar.header("Controls")
time_range = st.sidebar.selectbox("Select Time Range", ["Last 7 Days", "Last 14 Days", "Last 30 Days", "All"], index=3)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)")

# Load data
data = load_data_from_logs("app.log")

if data is None:
    st.error("No data to display. Please check your logs or run the model to generate logs.")
    st.stop()

# Filter data by time range
if time_range == "Last 7 Days":
    data_slice = slice(-7, None)
elif time_range == "Last 14 Days":
    data_slice = slice(-14, None)
elif time_range == "Last 30 Days":
    data_slice = slice(-30, None)
else:
    data_slice = slice(None)

dates = data['dates'][data_slice]
daily_volume = data['daily_volume'][data_slice]
errors = data['errors'][data_slice]
f1 = data['f1'][data_slice]
accuracy = data['accuracy'][data_slice]
roc_auc = data['roc_auc'][data_slice]
drift_scores = data['drift_scores'][data_slice]

# Key Metrics
st.markdown("---")
st.markdown("### Key Metrics")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    avg_f1score = np.nanmean(f1)
    st.metric("Average F1-Score", f"{avg_f1score:.3f}")

with col2:
    avg_accuracy = np.nanmean(accuracy)
    st.metric("Average Accuracy", f"{avg_accuracy:.3f}")

with col3:
    avg_auc = np.nanmean(roc_auc)
    st.metric("Average ROC AUC", f"{avg_auc:.3f}")

with col4:
    total_predictions = np.nansum(daily_volume)
    st.metric("Total Predictions", f"{int(total_predictions):,}")

with col5:
    total_errors = np.nansum(errors)
    st.metric("Total Errors", f"{int(total_errors)}")

with col6:
    avg_drift = np.nanmean(drift_scores)
    drift_status = "Normal" if avg_drift < 0.5 else "Warning" if avg_drift < 0.7 else "Critical"
    st.metric("Avg Data Drift Status", f"{drift_status} ({avg_drift:.2f})")

# Prediction Volume Plot
st.markdown("---")
st.subheader("Prediction Volume Over Time")
fig_volume = px.line(x=dates, y=daily_volume, labels={'x': 'Date', 'y': 'Predictions'})
st.plotly_chart(fig_volume, use_container_width=True)

# Errors Over Time
st.subheader("Errors Over Time")
fig_errors = px.bar(x=dates, y=errors, labels={'x': 'Date', 'y': 'Errors'}, color=errors, color_continuous_scale='Reds')
st.plotly_chart(fig_errors, use_container_width=True)

# Performance Metrics Trends
st.markdown("---")
st.subheader("Model Performance Metrics Over Time")
fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=dates, y=f1, mode='lines+markers', name='F1 Score'))
fig_perf.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='Accuracy'))
fig_perf.add_trace(go.Scatter(x=dates, y=roc_auc, mode='lines+markers', name='ROC AUC'))
fig_perf.update_layout(yaxis=dict(range=[0, 1]))
st.plotly_chart(fig_perf, use_container_width=True)

# Data Drift Over Time
st.markdown("---")
st.subheader("Data Drift Over Time")
fig_drift = px.line(x=dates, y=drift_scores, labels={'x': 'Date', 'y': 'Drift Score'})
fig_drift.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
fig_drift.add_hline(y=0.7, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
st.plotly_chart(fig_drift, use_container_width=True)

# Drift summary (last 7 days)
st.markdown("### Drift Summary (last 7 days)")
recent_drift = drift_scores[-7:]
normal_days = sum(x < 0.5 for x in recent_drift if not np.isnan(x))
warning_days = sum(0.5 <= x < 0.7 for x in recent_drift if not np.isnan(x))
critical_days = sum(x >= 0.7 for x in recent_drift if not np.isnan(x))
st.write(f"🟢 Normal days: {normal_days} / 7")
st.write(f"🟡 Warning days: {warning_days} / 7")
st.write(f"🔴 Critical days: {critical_days} / 7")

# if critical_days > 0:
#     st.warning("Critical drift detected! Consider retraining the model.")
# elif warning_days > 2:
#     st.info("Moderate drift observed. Monitor closely.")
# else:
#     st.success("Data drift is within acceptable limits.")

# 90-day drift counts
recent_drift_90 = drift_scores[-90:]
normal_90 = sum(x < 0.5 for x in recent_drift_90 if not np.isnan(x))
warning_90 = sum(0.5 <= x < 0.7 for x in recent_drift_90 if not np.isnan(x))
critical_90 = sum(x >= 0.7 for x in recent_drift_90 if not np.isnan(x))

st.markdown("**Drift Counts (Last 90 Days):**")
st.write(f"🟢 Normal: {normal_90}")
st.write(f"🟡 Warning: {warning_90}")
st.write(f"🔴 Critical: {critical_90}")

# 🚨 Drift warning logic for 90-day threshold
total_days_90 = len([x for x in recent_drift_90 if not np.isnan(x)])
warning_or_critical = warning_90 + critical_90

if total_days_90 > 0:
    drift_ratio = warning_or_critical / total_days_90
    if drift_ratio > 0.25:
        st.error(f"⚠️ High drift frequency detected! {warning_or_critical} out of {total_days_90} days "
                 f"({drift_ratio:.0%}) had warning or critical drift. Consider model review or retraining.")

# Download logs button
st.markdown("---")
st.subheader("Download Logs")
if os.path.exists("app.log"):
    with open("app.log", "rb") as f:
        st.download_button(label="Download app.log", data=f, file_name="app.log", mime="text/plain")
else:
    st.info("No log file found to download.")

# Auto-refresh every 30 seconds if enabled
if auto_refresh:
    st.experimental_rerun()
