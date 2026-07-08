import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.data_preprocessing import load_data, preprocess_data


def train_model():

    data = load_data("data/customer_churn.csv")

    data = preprocess_data(data)

    X = data.drop("Churn", axis=1)

    y = data["Churn"]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestClassifier(random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))

    joblib.dump(model, "models/churn_model.pkl")