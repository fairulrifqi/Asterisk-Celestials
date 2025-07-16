import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
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

# --- Custom Transformer ---
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None): return self
    def transform(self, x): return np.log1p(x)

# --- Feature Groups ---
num_columns = ['Age', 'NumCompaniesWorked', 'TotalWorkingYears',
               'TrainingTimesLastYear', 'YearsSinceLastPromotion',
               'YearsWithCurrManager', 'AvgWorkingHours']
ordinal_cat_columns = ['BusinessTravel', 'JobLevel', 'EnvironmentSatisfaction',
                       'JobSatisfaction', 'WorkLifeBalance']
ohe_columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
features = num_columns + ordinal_cat_columns + ohe_columns

# --- Preprocessor ---
def build_preprocessor():
    ordinal_encoder = OrdinalEncoder(categories=[
        ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],  # BusinessTravel
        [1, 2, 3, 4, 5],                                        # JobLevel
        [1, 2, 3, 4],                                           # EnvironmentSatisfaction
        [1, 2, 3, 4],                                           # JobSatisfaction
        [1, 2, 3, 4]                                            # WorkLifeBalance
    ])

    ohe_encoder = OneHotEncoder(categories=[
        ["Sales", "Research & Development", "Human Resources"],  # Department
        ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"],  # EducationField
        ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
         "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"],  # JobRole
        ["Single", "Married", "Divorced"]  # MaritalStatus
    ], drop='first', handle_unknown='ignore')

    num_pipeline = Pipeline([
        ('log', LogTransformer()),
        ('scaler', RobustScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_columns),
        ('ordinal', ordinal_encoder, ordinal_cat_columns),
        ('ohe', ohe_encoder, ohe_columns)
    ])
    return preprocessor

# --- Risk label function ---
def assign_risk_label(prob):
    if prob < 0.25:
        return "🟢 Safe"
    elif prob < 0.5:
        return "🟡 Monitor"
    elif prob < 0.75:
        return "🟠 At Risk"
    else:
        return "🔴 Likely Leave"

# --- UI ---
st.title("🔮 Employee Attrition Predictor")
st.write("by: Asterisk Celestials | Team 5")

tab1, tab2 = st.tabs(["🧍🧍🧍  Group of Employees", "🧍 Single Employee"])

# === TAB 1: Batch Upload ===
with tab1:
    st.markdown("### 1. Get The Data Template")
    st.write("⚠️ Read the instructions in the template file before you input your data ⚠️")
    with open("User_Template.xlsx", "rb") as file:
        st.download_button(label="📥 Download Template",
                           data=file,
                           file_name="input_template.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("### 2. Upload The Data")
    uploaded_file = st.file_uploader("Upload the input data", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
            st.write("📝 Uploaded data:")
            st.dataframe(df_input)

            if st.button("🔮 Predict"):
                # Fill missing values
                for col in num_columns:
                    df_input[col].fillna(df_input[col].median(), inplace=True)
                for col in ordinal_cat_columns + ohe_columns:
                    df_input[col].fillna(df_input[col].mode(dropna=True)[0], inplace=True)

                preprocessor = build_preprocessor()
                processed_data = preprocessor.fit_transform(df_input[features])
                prediction = model.predict(processed_data)
                prediction_proba = model.predict_proba(processed_data)[:, 1]

                df_input["Prediction"] = np.where(prediction == 1, "Leave", "Stay")
                df_input["Probability"] = prediction_proba
                df_input["Risk Level"] = df_input["Probability"].apply(assign_risk_label)

                st.markdown("### 📊 Prediction Results")
                st.dataframe(df_input)

                csv_result = df_input.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Results", csv_result, "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            logger.exception("Batch prediction error")
            st.error(f"An error occurred: {e}")

# === TAB 2: Single Input ===
with tab2:
    st.markdown("### Tell us about the employee!")
    with st.form("single_input_form"):
        employee_id = st.text_input("Employee ID")
        age = st.number_input("Age", 18, 65)
        num_companies = st.number_input("Number of Companies Worked", 0, 20)
        total_working_years = st.number_input("Total Working Years", 0, 50)
        training_times = st.number_input("Training Times Last Year", 0, 10)
        years_since_promotion = st.number_input("Years Since Last Promotion", 0, 20)
        years_with_manager = st.number_input("Years With Current Manager", 0, 20)
        avg_hours = st.number_input("Average Working Hours", min_value=0.0, max_value=24.0, step=0.1, format="%.1f")
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Other", "Medical", "Marketing", "Technical Degree", "Human Resources"])
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
                                             "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

        submit = st.form_submit_button("🔮 Predict")

    if submit:
        try:
            df_input = pd.DataFrame([{
                "EmployeeID": employee_id,
                "Age": age,
                "NumCompaniesWorked": num_companies,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times,
                "YearsSinceLastPromotion": years_since_promotion,
                "YearsWithCurrManager": years_with_manager,
                "AvgWorkingHours": avg_hours,
                "BusinessTravel": business_travel,
                "JobLevel": job_level,
                "EnvironmentSatisfaction": env_satisfaction,
                "JobSatisfaction": job_satisfaction,
                "WorkLifeBalance": work_life_balance,
                "Department": department,
                "EducationField": education_field,
                "JobRole": job_role,
                "MaritalStatus": marital_status
            }])

            preprocessor = build_preprocessor()
            processed_data = preprocessor.fit_transform(df_input[features])
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)[:, 1][0]

            label = "Leave" if prediction[0] == 1 else "Stay"
            risk_label = assign_risk_label(prediction_proba)

            st.success(f"Prediction: **{label}** with probability **{prediction_proba:.2%}** ({risk_label})")

        except Exception as e:
            logger.exception("Single input prediction failed")
            st.error(f"Prediction failed: {e}")
