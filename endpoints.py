import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="ML Models API")

# =========================
# LOAD MODELS
# =========================
with open("models/knn.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("models/linear_regression.pkl", "rb") as f:
    model1 = pickle.load(f)

with open("models/logistic_regression.pkl", "rb") as f:
    model2 = pickle.load(f)


# =========================
# REQUEST SCHEMA
# =========================
class PredictionInput(BaseModel):
    features: list[float]


# =========================
# ROOT
# =========================
@app.get("/")
def home():
    return {"message": "ML Models API is running"}


# =========================
# KNN PREDICTION
# =========================
@app.post("/predict/knn")
def predict_knn(data: PredictionInput):

    features = np.array(data.features).reshape(1, -1)

    # safety check
    if features.shape[1] != knn_model.n_features_in_:
        return {
            "error": f"Expected {knn_model.n_features_in_} features, got {features.shape[1]}"
        }

    prediction = knn_model.predict(features)[0]

    return {
        "model": "KNN",
        "prediction": int(prediction)
    }


# =========================
# LINEAR REGRESSION
# =========================
@app.post("/predict/linear_regression")
def predict_linear_regression(data: PredictionInput):

    features = np.array(data.features).reshape(1, -1)

    if features.shape[1] != model1.n_features_in_:
        return {
            "error": f"Expected {model1.n_features_in_} features, got {features.shape[1]}"
        }

    prediction = model1.predict(features)[0]

    return {
        "model": "Linear Regression",
        "prediction": float(prediction)
    }


# =========================
# LOGISTIC REGRESSION
# =========================
@app.post("/predict/logistic_regression")
def predict_logistic_regression(data: PredictionInput):

    features = np.array(data.features).reshape(1, -1)

    if features.shape[1] != model2.n_features_in_:
        return {
            "error": f"Expected {model2.n_features_in_} features, got {features.shape[1]}"
        }

    prediction = model2.predict(features)[0]

    return {
        "model": "Logistic Regression",
        "prediction": int(prediction)
    }