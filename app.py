import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load("heart_disease_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("❤️ Heart Disease Prediction")
st.write(
    "Enter the patient's medical information "
    "to predict the possibility of heart disease."
)


# -----------------------------
# Input Fields
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.selectbox(
    "Resting ECG",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    options=[0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    options=[0, 1, 2, 3]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict Heart Disease"):

    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=feature_names
    )

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # -----------------------------
    # Display Result
    # -----------------------------

    if prediction == 1:

        st.error(
            "⚠️ Prediction: Possible Heart Disease"
        )

        st.write(
            f"Estimated probability: "
            f"{probability * 100:.2f}%"
        )

    else:

        st.success(
            "✅ Prediction: No Heart Disease Detected"
        )

        st.write(
            f"Estimated probability: "
            f"{probability * 100:.2f}%"
        )


# -----------------------------
# Disclaimer
# -----------------------------

st.warning(
    "This application is for educational purposes only "
    "and should not be used as a substitute for professional medical advice."
)