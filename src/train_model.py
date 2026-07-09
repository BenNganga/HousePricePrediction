import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from src.data_preprocessing import load_data, preprocess_data


def train_model():

    data = load_data("data/housing.csv")

    data = preprocess_data(data)

    X = data.drop("Price", axis=1)

    y = data["Price"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Mean Absolute Error:",
          mean_absolute_error(y_test, predictions))

    joblib.dump(model, "models/house_price_model.pkl")