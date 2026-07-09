import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    layout="centered"
)

st.title("🏠 House Price Prediction")

model = joblib.load("models/house_price_model.pkl")

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

area = st.number_input(
    "Area (Square Feet)",
    min_value=500,
    max_value=10000,
    value=2000
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=2
)

if st.button("Predict Price"):

    sample = pd.DataFrame({

        "Bedrooms":[bedrooms],

        "Bathrooms":[bathrooms],

        "Area":[area],

        "Parking":[parking]

    })

    prediction = model.predict(sample)

    st.success(f"Estimated House Price: ${prediction[0]:,.2f}")