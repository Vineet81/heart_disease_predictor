# ❤️ Heart Disease Predictor

A machine learning web app built with **Streamlit** that predicts the likelihood of heart disease using patient data.

---

## 🔍 Project Overview

This app uses two classification algorithms:
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

The models are trained on a cleaned heart disease dataset, and predictions are made based on features such as:
- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Rest ECG
- Max Heart Rate
- Exercise Induced Angina
- Oldpeak
- Slope, etc.

---

## 🚀 Features

- Simple UI built with Streamlit
- Real-time heart disease prediction
- Uses ColumnTransformer + OneHotEncoder for categorical features
- Deployed on Streamlit Cloud

---

## 🧠 Models Used

| Model                | Technique           |
|---------------------|---------------------|
| Support Vector Classifier | Classification |
| Random Forest Classifier | Classification |

---

## 📦 Dependencies

Install requirements with:

```bash
pip install -r requirements.txt
How to Run Locally

    Clone the repo:

git clone https://github.com/YOUR_USERNAME/heart_disease_predictor.git
cd heart_disease_predictor

    Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies:

pip install -r requirements.txt

    Run the app:

streamlit run app.py

🧠 Model Training

The model_training.ipynb notebook contains code for:

    Exploratory Data Analysis

    Preprocessing (with ColumnTransformer)

    Training using SVM and RandomForest

    Saving models and encoders with joblib

📁 File Structure

heart_disease_predictor/
│
├── app.py                  # Streamlit app
├── model_training.ipynb    # Notebook for training
├── heart.csv               # Dataset
├── svm_model.pkl           # Trained SVM model
├── rf_model.pkl            # Trained Random Forest model
├── preprocessor.pkl        # Preprocessing pipeline
├── requirements.txt
├── .gitignore
└── README.md

🌐 Deployment

You can deploy this app easily on:

    Streamlit Cloud

    Hugging Face Spaces (for advanced ML hosting)

    Docker (optional)

📧 Contact

For queries or collaboration, feel free to connect on GitHub.
