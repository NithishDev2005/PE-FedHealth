from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "stress_model_bundle.pkl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# LOAD MODEL
# =========================================================

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["model"]

scaler = bundle["scaler"]

background = np.asarray(
    bundle["background"]
)


FEATURES = [
    "ecg_std",
    "eda_mean",
    "temp_std",
    "resp_std"
]


# =========================================================
# SCALE BACKGROUND
# =========================================================

background_scaled = (
    scaler.transform(
        background
    )
)


# =========================================================
# SHAP EXPLAINER
# =========================================================

explainer = shap.LinearExplainer(
    model,
    background_scaled,
    feature_names=FEATURES
)


# =========================================================
# LOAD DATA
# =========================================================

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_features.csv"
)

df = pd.read_csv(
    DATA_PATH
)


df = df[
    df["label"].isin([1, 2])
].copy()


# =========================================================
# SAMPLE
# =========================================================

sample = df[
    FEATURES
].iloc[
    0:1
].values


sample_scaled = scaler.transform(
    sample
)


# =========================================================
# SHAP VALUES
# =========================================================

shap_values = explainer(
    sample_scaled
)


values = (
    shap_values.values[0]
)


print()
print("==========================================")
print("          SHAP EXPLANATION")
print("==========================================")


for (
    feature,
    value
) in zip(
    FEATURES,
    values
):

    print(
        f"{feature}: "
        f"{float(value):.4f}"
    )


# =========================================================
# SAVE CSV
# =========================================================

result = pd.DataFrame({
    "feature": FEATURES,
    "shap_value": [
        float(value)
        for value in values
    ]
})


output_path = (
    OUTPUT_DIR
    / "shap_values.csv"
)


result.to_csv(
    output_path,
    index=False
)


print()
print(
    "SHAP results saved to:"
)

print(
    output_path
)

print()
print("==========================================")