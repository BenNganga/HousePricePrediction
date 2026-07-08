import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction System")

model = joblib.load("models/churn_model.pkl")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=50.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=500.0
)

if st.button("Predict"):

    data = pd.DataFrame({

        "gender":[gender],

        "SeniorCitizen":[senior],

        "tenure":[tenure],

        "MonthlyCharges":[monthly],

        "TotalCharges":[total]

    })

    data = pd.get_dummies(data)

    prediction = model.predict(data)

    if prediction[0] == 1:

        st.error("Customer is likely to Churn.")

    else:

        st.success("Customer is likely to Stay.")