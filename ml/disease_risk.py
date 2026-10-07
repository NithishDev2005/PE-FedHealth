from pathlib import Path
import json

import joblib
import numpy as np


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "stress_model_bundle.pkl"
)

METRICS_PATH = (
    BASE_DIR
    / "models"
    / "stress_model_metrics.json"
)


# =========================================================
# LOAD MODEL
# =========================================================

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["model"]

scaler = bundle["scaler"]

FEATURES = bundle.get(
    "features",
    [
        "ecg_std",
        "eda_mean",
        "temp_std",
        "resp_std"
    ]
)


# =========================================================
# LOAD METRICS
# =========================================================

try:

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        MODEL_METRICS = json.load(
            file
        )

except Exception:

    MODEL_METRICS = {}


# =========================================================
# RISK CALCULATION
# =========================================================

def calculate_risk(features):

    values = np.array([
        features["ecg_std"],
        features["eda_mean"],
        features["temp_std"],
        features["resp_std"]
    ]).reshape(1, -1)


    scaled_values = scaler.transform(
        values
    )


    probabilities = model.predict_proba(
        scaled_values
    )[0]


    stress_probability = float(
        probabilities[1]
    )


    prediction = model.predict(
        scaled_values
    )[0]


    predicted_state = (
        "Stress"
        if prediction == 1
        else "Baseline"
    )


    # =====================================================
    # SIGNAL-BASED INDICATORS
    # =====================================================

    cardiac_indicator = min(
        abs(features["ecg_std"]) * 50,
        100
    )


    respiratory_indicator = min(
        abs(features["resp_std"]) * 50,
        100
    )


    stress_risk = (
        stress_probability * 100
    )


    overall_risk = (
        0.70 * stress_risk
        +
        0.15 * cardiac_indicator
        +
        0.15 * respiratory_indicator
    )


    return {

        "cardiac_risk": round(
            float(cardiac_indicator),
            1
        ),

        "stress_risk": round(
            float(stress_risk),
            1
        ),

        "respiratory_risk": round(
            float(respiratory_indicator),
            1
        ),

        "overall_risk": round(
            float(overall_risk),
            1
        ),

        "predicted_state":
            predicted_state,

        "stress_probability": round(
            stress_probability,
            4
        ),

        "confidence": round(
            max(probabilities),
            4
        ),

        "model_type":
            "Logistic Regression",

        "task":
            "Baseline vs Stress",

        "model_metrics":
            MODEL_METRICS,

        "note": (
            "These are research-oriented "
            "physiological indicators and "
            "are not clinical diagnoses."
        )
    }