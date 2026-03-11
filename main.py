# =============================================================================
# IMPORTS & APP SETUP
# =============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# NEW: explainability imports
import shap
from lime import lime_tabular

# Create FastAPI app
app = FastAPI(title="Wine ML API", version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# LOAD MODELS
# =============================================================================

fnn_model = tf.keras.models.load_model("models/fnn_quality_classifier.h5")

with open("models/fnn_scaler.pkl", "rb") as f:
    fnn_scaler = pickle.load(f)

# =============================================================================
# LOAD TRAINING DATA (for SHAP/LIME background)
# =============================================================================

X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")["quality"]

# =============================================================================
# COUNTRY ONE-HOT COLUMNS
# =============================================================================

COUNTRY_COLUMNS = [
    'country_Armenia', 'country_Australia', 'country_Austria',
    'country_Bosnia and Herzegovina', 'country_Brazil', 'country_Bulgaria',
    'country_Canada', 'country_Chile', 'country_China', 'country_Croatia',
    'country_Cyprus', 'country_Czech Republic', 'country_England',
    'country_France', 'country_Georgia', 'country_Germany', 'country_Greece',
    'country_Hungary', 'country_India', 'country_Israel', 'country_Italy',
    'country_Lebanon', 'country_Luxembourg', 'country_Macedonia',
    'country_Mexico', 'country_Moldova', 'country_Morocco',
    'country_New Zealand', 'country_Peru', 'country_Portugal',
    'country_Romania', 'country_Serbia', 'country_Slovenia',
    'country_South Africa', 'country_Spain', 'country_Switzerland',
    'country_Turkey', 'country_US', 'country_Ukraine', 'country_Uruguay'
]

FEATURE_HEADERS = ["log_price", "clean_length", "sentiment_num"] + COUNTRY_COLUMNS

QUALITY_MAP = {0: "Low", 1: "Medium", 2: "High"}

# =============================================================================
# REQUEST MODELS
# =============================================================================

class FNNRequest(BaseModel):
    log_price: float
    clean_length: float
    sentiment_num: int
    country: str

class FNNRequestBatch(BaseModel):
    items: List[FNNRequest]

class FeatureImportanceResponse(BaseModel):
    features: List[str]
    importances: List[float]

# =============================================================================
# REUSABLE PREPROCESSING FUNCTION FOR FNN
# =============================================================================

def preprocess_fnn_input(req: FNNRequest):
    """
    Converts raw API input into the 43‑dimensional scaled model input.
    Ensures identical preprocessing for prediction, uncertainty, SHAP, and LIME.
    """
    # 1. Numeric features
    numeric = np.array([[req.log_price, req.clean_length, req.sentiment_num]], dtype=np.float32)
    numeric_scaled = fnn_scaler.transform(numeric)[0]   # shape (3,)

    # 2. Country one‑hot
    country_vector = np.zeros(len(COUNTRY_COLUMNS), dtype=np.float32)
    country_key = f"country_{req.country}"

    if country_key in COUNTRY_COLUMNS:
        idx = COUNTRY_COLUMNS.index(country_key)
        country_vector[idx] = 1.0

    # 3. Combine into final 43‑dimensional vector
    final_input = np.concatenate([numeric_scaled, country_vector]).astype(np.float32)

    return final_input.reshape(1, -1)   # shape (1, 43)

# =============================================================================
# MONTE CARLO DROPOUT FOR UNCERTAINTY
# =============================================================================

def mc_dropout_predict(model, x, n_samples=50):
    preds = []
    for _ in range(n_samples):
        p = model(x, training=True).numpy()
        preds.append(p)

    preds = np.array(preds)
    mean_probs = preds.mean(axis=0)[0]
    std_probs = preds.std(axis=0)[0]

    pred_class = int(np.argmax(mean_probs))
    uncertainty = float(std_probs[pred_class])

    return {
        "mean_probs": mean_probs.tolist(),
        "std_probs": std_probs.tolist(),
        "pred_class": pred_class,
        "uncertainty": uncertainty
    }

# =============================================================================
# ROOT ENDPOINT
# =============================================================================

@app.get("/")
def home():
    return {"status": "Wine ML API is running"}

# =============================================================================
# FNN SINGLE PREDICTION ENDPOINT
# =============================================================================

@app.post("/predict_fnn")
def predict_fnn(request: FNNRequest):

    x = preprocess_fnn_input(request)
    pred = fnn_model.predict(x)
    probabilities = pred[0]

    pred_class = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))

    return {
        "model": "FNN Quality Classifier",
        "prediction": pred_class,
        "confidence": confidence
    }

# =============================================================================
# FNN BATCH PREDICTION ENDPOINT
# =============================================================================

@app.post("/predict_fnn_batch")
def predict_fnn_batch(request: FNNRequestBatch):

    predictions = []

    for item in request.items:
        x = preprocess_fnn_input(item)
        pred = fnn_model.predict(x)
        pred_class = int(np.argmax(pred, axis=1)[0])
        predictions.append(pred_class)

    return {
        "model": "FNN Quality Classifier (Batch)",
        "predictions": predictions
    }

# =============================================================================
# FNN UNCERTAINTY ENDPOINT (MC DROPOUT)
# =============================================================================

@app.post("/predict_fnn_uncertainty")
def predict_fnn_uncertainty(input: FNNRequest):

    x = preprocess_fnn_input(input)

    mc = mc_dropout_predict(fnn_model, x, n_samples=50)
    mean_probs = np.array(mc["mean_probs"])
    std_probs = np.array(mc["std_probs"])

    pred_class = int(np.argmax(mean_probs))
    pred_label = QUALITY_MAP[pred_class]

    u = float(std_probs[pred_class])
    if u < 0.05:
        u_label = "Low"
    elif u < 0.12:
        u_label = "Moderate"
    else:
        u_label = "High"

    uncertainty_str = f"({u:.2f}) {u_label}"

    top_two = mean_probs.argsort()[-2:]
    low_val = float(mean_probs[top_two[0]])
    high_val = float(mean_probs[top_two[1]])

    ci_unc = max(float(std_probs[top_two[0]]), float(std_probs[top_two[1]]))

    if ci_unc < 0.05:
        ci_label = "Low"
    elif ci_unc < 0.12:
        ci_label = "Moderate"
    else:
        ci_label = "High"

    ci_str = f"({low_val:.2f} ↔ {high_val:.2f}) {ci_label}"

    return {
        "predicted_quality": pred_label,
        "uncertainty": uncertainty_str,
        "confidence_interval": ci_str,
        "technical": {
            "mean_probs": mean_probs.tolist(),
            "std_probs": std_probs.tolist(),
            "pred_class": pred_class,
            "uncertainty_raw": u
        }
    }

# =============================================================================
# FEATURE IMPORTANCE ENDPOINT (WEIGHT-BASED)
# =============================================================================

@app.get("/fnn_feature_importance", response_model=FeatureImportanceResponse)
def fnn_feature_importance():

    try:
        model = tf.keras.models.load_model("models/fnn_quality_classifier.h5")

        first_layer = model.layers[0]
        weights, biases = first_layer.get_weights()

        importance_scores = np.sum(np.abs(weights), axis=1)
        importance_scores = importance_scores / np.sum(importance_scores)

        feature_names = FEATURE_HEADERS

        return FeatureImportanceResponse(
            features=feature_names,
            importances=importance_scores.tolist()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SHAP EXPLAINABILITY ENDPOINT (SINGLE PREDICTION)
# =============================================================================

# Build a small background sample for SHAP (to keep it fast)
# Use model input space: X_train is already in the same 43‑dimensional space.
shap_background = X_train.sample(n=min(100, len(X_train)), random_state=42).values
shap_explainer = shap.KernelExplainer(fnn_model.predict, shap_background)

@app.post("/explain_shap_single")
def explain_shap_single(request: FNNRequest):
    """
    Returns SHAP values for a single FNN prediction.
    """
    try:
        x = preprocess_fnn_input(request)          # shape (1, 43)
        shap_values = shap_explainer.shap_values(x, nsamples=100)

        # shap_values is a list (one array per class) for softmax outputs
        # We'll return:
        # - per-class SHAP values
        # - base values
        # - predicted class
        pred = fnn_model.predict(x)[0]
        pred_class = int(np.argmax(pred))

        return {
            "feature_names": FEATURE_HEADERS,
            "predicted_class": pred_class,
            "predicted_probs": pred.tolist(),
            "base_values": [float(b) for b in shap_explainer.expected_value] \
                           if isinstance(shap_explainer.expected_value, (list, np.ndarray))
                           else [float(shap_explainer.expected_value)],
            "shap_values": [
                sv[0].tolist() for sv in shap_values  # one list per class
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LIME EXPLAINABILITY ENDPOINT (SINGLE PREDICTION)
# =============================================================================

# Build LIME explainer on training data
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=FEATURE_HEADERS,
    class_names=["Low", "Medium", "High"],
    mode="classification"
)

@app.post("/explain_lime_single")
def explain_lime_single(request: FNNRequest):
    """
    Returns LIME explanation for a single FNN prediction.
    """
    try:
        x = preprocess_fnn_input(request)          # shape (1, 43)
        x_row = x[0]

        exp = lime_explainer.explain_instance(
            data_row=x_row,
            predict_fn=fnn_model.predict,
            num_features=10
        )

        # exp.as_list() -> list of (feature_description, weight)
        explanation_items = exp.as_list()

        return {
            "feature_names": FEATURE_HEADERS,
            "explanation": [
                {
                    "feature": str(feat),
                    "weight": float(weight)
                }
                for feat, weight in explanation_items
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
