import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load("tuned_classifier.pkl")

model = load_model()

# --- Download Template ---
st.title("🧠 Machine Learning Model Deployment")

st.markdown("### 1. Unduh Template Input")
with open("User_Template.xlsx", "rb") as file:
    st.download_button(label="📥 Download Template",
                       data=file,
                       file_name="input_template.xlsx",
                       mime="text/csv")

# --- Upload File ---
st.markdown("### 2. Upload File Data")
uploaded_file = st.file_uploader("Unggah file CSV sesuai template", type=["csv", "xlsx"])

# --- Predict Button ---
if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.write("📝 Data yang diunggah:")
    st.dataframe(df_input)

    # Tombol prediksi
    if st.button("🔮 Predict"):
        try:
            # --- Pipeline preprocessing ---
            employee_ids = df_input['EmployeeID'].copy()

            # Fitur numerik dan kategorikal
            num_columns = ['Age', 'NumCompaniesWorked', 'TotalWorkingYears',
                           'TrainingTimesLastYear', 'YearsSinceLastPromotion',
                           'YearsWithCurrManager', 'AvgWorkingHours']
            ordinal_cat_columns = ['BusinessTravel', 'JobLevel', 'EnvironmentSatisfaction',
                                   'JobSatisfaction', 'WorkLifeBalance']
            ohe_columns = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

            # Custom Log Transformer
            class LogTransformer(BaseEstimator, TransformerMixin):
                def fit(self, x, y=None): return self
                def transform(self, x): return np.log1p(x)

            num_pipeline = Pipeline([
                ('log', LogTransformer()),
                ('scaler', RobustScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, num_columns),
                ('ordinal', OrdinalEncoder(categories=[
                    sorted(df_input[col].dropna().unique().tolist()) for col in ordinal_cat_columns
                ]), ordinal_cat_columns),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'), ohe_columns)
            ])

            # Proses data
            features_to_process = num_columns + ordinal_cat_columns + ohe_columns
            processed_data = preprocessor.fit_transform(df_input[features_to_process])

            # Prediksi
            prediction = model.predict(processed_data)

            # Tampilkan hasil
            st.markdown("### 📊 Hasil Prediksi")
            df_input["Prediction"] = prediction
            st.dataframe(df_input)

            # Unduh hasil
            csv_result = df_input.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Unduh Hasil Prediksi", csv_result, "hasil_prediksi.csv", "text/csv")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
