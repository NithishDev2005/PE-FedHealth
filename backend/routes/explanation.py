from pathlib import Path

import joblib
import numpy as np
import shap

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(
    prefix="/explanation",
    tags=["Explainability"]
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "stress_model_bundle.pkl"
)


# =========================================================
# FEATURES
# =========================================================

FEATURES = [
    "ecg_std",
    "eda_mean",
    "temp_std",
    "resp_std"
]


# =========================================================
# LOAD MODEL
# =========================================================

bundle = joblib.load(MODEL_PATH)

scaler = bundle["scaler"]
classifier = bundle["model"]

background = np.asarray(
    bundle["background"]
)

background_scaled = scaler.transform(
    background
)


# =========================================================
# SHAP
# =========================================================

explainer = shap.LinearExplainer(
    classifier,
    background_scaled,
    feature_names=FEATURES
)


# =========================================================
# INPUT
# =========================================================

class ExplanationInput(BaseModel):

    ecg_std: float
    eda_mean: float
    temp_std: float
    resp_std: float


# =========================================================
# EXPLANATION API
# =========================================================

@router.post("/shap")
def explain_prediction(data: ExplanationInput):

    values = np.array([
        data.ecg_std,
        data.eda_mean,
        data.temp_std,
        data.resp_std
    ]).reshape(1, -1)

    scaled_values = scaler.transform(
        values
    )

    shap_result = explainer(
        scaled_values
    )

    contributions = shap_result.values[0]

    probability = classifier.predict_proba(
        scaled_values
    )[0, 1]

    prediction = classifier.predict(
        scaled_values
    )[0]

    predicted_state = (
        "Stress"
        if prediction == 1
        else "Baseline"
    )

    expected_value = explainer.expected_value

    if isinstance(expected_value, np.ndarray):
        expected_value = expected_value.tolist()

    return {
        "feature_contributions": {
            FEATURES[index]: float(
                contributions[index]
            )
            for index in range(
                len(FEATURES)
            )
        },

        "predicted_state": predicted_state,

        "stress_probability": float(
            probability
        ),

        "expected_value": expected_value,

        "method": "SHAP LinearExplainer",

        "note": (
            "Feature contributions explain "
            "the model output and do not "
            "represent medical causation."
        )
    }