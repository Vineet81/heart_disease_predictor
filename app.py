# app.py — Streamlit UI for Heart Disease Prediction

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Load Pretrained Artifacts
preprocessor = joblib.load("preprocessor.pkl")
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")

# Define Feature Columns
categorical_cols = ["cp", "thal", "slope"]
numerical_cols = [
    'age', 'sex','trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'ca'
]

# UI Title
st.title("Heart Disease Prediction App")

# Sidebar Options
mode = st.sidebar.selectbox("Choose Mode", ["Single Prediction", "Bulk Prediction", "Visualization"])
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "SVM"])

model = rf_model if model_choice == "Random Forest" else svm_model

# Input Function for Single Prediction
def get_single_input():
    user_data = {}
    for col in numerical_cols:
        user_data[col] = st.number_input(f"Enter {col}", value=0.0 if col != 'ca' else 0, format="%f")

    user_data['cp'] = st.selectbox("Chest Pain Type", ["typical_angina", "atypical_angina", "non_anginal_pain", "asymptomatic"])
    user_data['thal'] = st.selectbox("Thalassemia", ["normal", "fixed_defect", "reversible_defect"])
    user_data['slope'] = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])
    
    return pd.DataFrame([user_data])

# Mode Logic
if mode == "Single Prediction":
    st.subheader("Predict Heart Disease - Single Input")
    input_df = get_single_input()
    if st.button("Predict"):
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]
        st.success("Prediction: {}".format("Heart Disease" if prediction == 1 else "No Heart Disease"))

elif mode == "Bulk Prediction":
    st.subheader("Bulk Prediction from CSV")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        X_processed = preprocessor.transform(df)
        predictions = model.predict(X_processed)
        df['Prediction'] = predictions
        df['Prediction'] = df['Prediction'].map({1: "Heart Disease", 0: "No Heart Disease"})
        st.write(df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

elif mode == "Visualization":
    st.subheader("Explore Dataset")
    file = st.file_uploader("Upload Dataset for Visualization", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write(df.head())
        target_col = st.selectbox("Select Target Column for Analysis", df.columns)
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=target_col, ax=ax)
        st.pyplot(fig)

# End of app.py

