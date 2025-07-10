import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Load model ---
@st.cache_resource
def load_model():
    logger.info("Loading model from tuned_classifier.pkl...")
    model = joblib.load("tuned_classifier.pkl")
    logger.info("Model loaded successfully.")
    return model

model = load_model()

# --- UI Title and Template Download ---
st.title("🔮 Employee Attrition Predictor")
st.write("by: Asterisk Celestials | Team 5")

st.markdown("### Get The Data Template")
with open("User_Template.xlsx", "rb") as file:
    st.write("⚠️ Read the instructions in the template file before you input your data ⚠️")
    st.download_button(label="📥 Download Template",
                       data=file,
                       file_name="input_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Upload File ---
st.markdown("### 2. Upload The Data")
uploaded_file = st.file_uploader("Upload the input data", type=["csv", "xlsx"])

if uploaded_file:
    logger.info(f"File uploaded: {uploaded_file.name}")

    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df_input = pd.read_csv(uploaded_file)
        else:
            df_input = pd.read_excel(uploaded_file)

        st.write("📝 Uploaded data:")
        st.dataframe(df_input)

        # --- Predict Button ---
        if st.button("🔮 Predict"):
            logger.info("Prediction made")  # For monitoring count

            # Keep a copy of IDs
            employee_ids = df_input['EmployeeID'].copy()

            # --- Feature Groups ---
            num_columns = ['Age', 'NumCompaniesWorked', 'TotalWorkingYears',
                           'TrainingTimesLastYear', 'YearsSinceLastPromotion',
                           'YearsWithCurrManager', 'AvgWorkingHours']
            ordinal_cat_columns = ['BusinessTravel', 'JobLevel', 'EnvironmentSatisfaction',
                                   'JobSatisfaction', 'WorkLifeBalance']
            ohe_columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

            # --- Fill missing numeric with median ---
            for col in num_columns:
                if df_input[col].isnull().any():
                    median_val = df_input[col].median()
                    df_input[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing numeric values in '{col}' with median: {median_val}")

            # --- Fill missing categorical with mode ---
            for col in ordinal_cat_columns + ohe_columns:
                if df_input[col].isnull().any():
                    mode_val = df_input[col].mode(dropna=True)[0]
                    df_input[col].fillna(mode_val, inplace=True)
                    logger.info(f"Filled missing categorical values in '{col}' with mode: {mode_val}")

            # --- Custom Log Transformer ---
            class LogTransformer(BaseEstimator, TransformerMixin):
                def fit(self, x, y=None): return self
                def transform(self, x): return np.log1p(x)

            # --- Pipelines ---
            num_pipeline = Pipeline([
                ('log', LogTransformer()),
                ('scaler', RobustScaler())
            ])

            ordinal_encoder = OrdinalEncoder(categories=[
                sorted(df_input[col].unique().tolist()) for col in ordinal_cat_columns
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_columns),
                ('ordinal', ordinal_encoder, ordinal_cat_columns),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_columns)
            ])

            features = num_columns + ordinal_cat_columns + ohe_columns
            processed_data = preprocessor.fit_transform(df_input[features])
            logger.info("Data preprocessing complete.")

            # --- Predict ---
            prediction = model.predict(processed_data)
            logger.info(f"Prediction complete. Total predictions: {len(prediction)}")

            # Count predictions
            pred_counts = pd.Series(prediction).value_counts().to_dict()
            label_map = {0: "Stay", 1: "Leave"}
            pred_readable = {label_map.get(k, str(k)): v for k, v in pred_counts.items()}
            logger.info(f"Prediction results: {pred_readable}")

            # --- Simulated Metrics (for monitoring) ---
            f1 = round(0.80 + np.random.uniform(-0.02, 0.02), 4)
            acc = round(0.85 + np.random.uniform(-0.02, 0.02), 4)
            auc = round(0.88 + np.random.uniform(-0.02, 0.02), 4)
            drift = round(np.random.uniform(0.3, 0.6), 4)

            logger.info(f"METRICS | f1: {f1} | accuracy: {acc} | roc_auc: {auc} | drift_score: {drift}")

            # --- Prediction Probability ---
            prediction_proba = model.predict_proba(processed_data)[:, 1]
            # --- Show result ---
            df_input["Prediction"] = np.where(prediction == 1, "Leave", "Stay")
            df_input["Probability"] = prediction_proba

            st.markdown("### 📊 The Prediction is Here")
            st.dataframe(df_input)

            # --- Download result ---
            csv_result = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Get the result", csv_result, "hasil_prediksi.csv", "text/csv")

    except Exception as e:
        logger.exception("ERROR during prediction")
        st.error(f"Errors occured during prediction proses: {e}")
