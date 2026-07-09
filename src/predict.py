import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")


def predict(data):

    sample = pd.DataFrame([data])

    sample = pd.get_dummies(sample)

    prediction = model.predict(sample)

    return prediction[0]