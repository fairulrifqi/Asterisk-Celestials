import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset, DataDriftPreset
import streamlit.components.v1 as components

st.set_page_config("Attrition Monitoring", layout="wide")

# 1. Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")  # ganti sesuai path model

model = load_model()

# 2. Upload data untuk prediksi
st.sidebar.title("🔍 Prediction Input")
upload = st.sidebar.file_uploader("Upload data (CSV)", type=["csv"])

if upload:
    df = pd.read_csv(upload)
    st.write("## 📊 Uploaded Data", df.head())

    # 3. Prediksi lokal
    if st.sidebar.button("Run Prediction"):
        X = df.drop(columns=["Attrition"], errors='ignore')  # asumsi kolom target bernama 'Attrition'
        y_true = df["Attrition"] if "Attrition" in df.columns else None

        y_pred = model.predict(X)
        df["prediction"] = y_pred
        if y_true is not None:
            df["true_label"] = y_true

        df["timestamp"] = datetime.now()

        # 4. Simpan ke log file
        df.to_csv("logs/prediction_log.csv", mode='a', header=not pd.io.common.file_exists("logs/prediction_log.csv"), index=False)
        st.success("✅ Prediction logged successfully!")

        st.write("## 🔮 Prediction Results", df.head())

# 5. Monitoring dan Metrik
st.markdown("---")
st.header("📈 Model Monitoring")

@st.cache_data
def load_logs():
    try:
        df_log = pd.read_csv("logs/prediction_log.csv", parse_dates=["timestamp"])
        df_log["date"] = df_log["timestamp"].dt.date
        return df_log
    except:
        return pd.DataFrame()

log_df = load_logs()
if log_df.empty:
    st.warning("Belum ada data log.")
else:
    st.write("### 🔁 Logged Predictions", log_df.tail())

    # 6. Agregasi metrik harian
    daily = log_df.dropna(subset=["true_label"]).groupby("date").agg({
        "prediction": lambda x: list(x),
        "true_label": lambda x: list(x)
    }).reset_index()

    def calc_metrics(row):
        return pd.Series({
            "f1": f1_score(row["true_label"], row["prediction"]),
            "accuracy": accuracy_score(row["true_label"], row["prediction"]),
            "roc_auc": roc_auc_score(row["true_label"], row["prediction"])
        })

    metrics_df = daily.join(daily.apply(calc_metrics, axis=1))

    st.write("### 📊 Daily Metrics")
    st.line_chart(metrics_df.set_index("date")[["f1", "accuracy", "roc_auc"]])

# 7. Evidently Report
st.markdown("---")
st.header("🧪 Evidently Monitoring Report")

if not log_df.empty and "true_label" in log_df.columns:
    # Load reference dataset
    reference = pd.read_csv("reference_with_labels.csv")  # pastikan kolom sama

    # Jalankan report
    report = Report(metrics=[ClassificationPreset(), DataDriftPreset()])
    report.run(reference_data=reference, current_data=log_df)

    report.save_html("evidently_report.html")

    with open("evidently_report.html", "r", encoding="utf-8") as f:
        html = f.read()
        components.html(html, height=800, scrolling=True)
else:
    st.info("Tambahkan data dengan label 'true_label' untuk Evidently.")
