from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_features.csv"
)

MODEL_DIR = (
    BASE_DIR
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    MODEL_DIR
    / "stress_model_bundle.pkl"
)

METRICS_PATH = (
    MODEL_DIR
    / "stress_model_metrics.json"
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
# LOAD DATA
# =========================================================

df = pd.read_csv(DATA_PATH)

print()
print("==========================================")
print("       PE-FedHealth ML TRAINING")
print("==========================================")

print(
    "Dataset shape:",
    df.shape
)


# =========================================================
# BASELINE VS STRESS
# =========================================================

df = df[
    df["label"].isin([1, 2])
].copy()

df["target"] = (
    df["label"]
    .map({
        1: 0,
        2: 1
    })
)


# =========================================================
# REMOVE INVALID ROWS
# =========================================================

df = df.dropna(
    subset=FEATURES + [
        "target",
        "subject"
    ]
)


X = df[FEATURES]

y = df["target"]

groups = df["subject"]


print(
    "Filtered samples:",
    len(df)
)

print(
    "Classes:",
    sorted(y.unique().tolist())
)


# =========================================================
# SUBJECT-AWARE SPLIT
# =========================================================

splitter = GroupShuffleSplit(
    n_splits=20,
    test_size=0.20,
    random_state=42
)

train_idx = None
test_idx = None

for current_train, current_test in splitter.split(
    X,
    y,
    groups
):

    train_classes = set(
        y.iloc[current_train]
    )

    test_classes = set(
        y.iloc[current_test]
    )

    if (
        len(train_classes) == 2
        and len(test_classes) == 2
    ):

        train_idx = current_train
        test_idx = current_test

        break


if train_idx is None:

    raise RuntimeError(
        "Could not create a subject-aware "
        "train/test split containing both classes."
    )


X_train = X.iloc[train_idx]

X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]

y_test = y.iloc[test_idx]


# =========================================================
# MODEL
# =========================================================

pipeline = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# =========================================================
# TRAIN
# =========================================================

print()
print("Training Logistic Regression...")

pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# EVALUATION
# =========================================================

predictions = pipeline.predict(
    X_test
)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print()
print("==========================================")
print("MODEL EVALUATION")
print("==========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# =========================================================
# FINAL MODEL
# =========================================================

final_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

final_model.fit(
    X,
    y
)


# =========================================================
# EXTRACT MODEL COMPONENTS
# =========================================================

final_scaler = final_model.named_steps[
    "scaler"
]

final_classifier = final_model.named_steps[
    "model"
]


# =========================================================
# SHAP BACKGROUND
# =========================================================

background_size = min(
    100,
    len(X)
)

background = X.sample(
    n=background_size,
    random_state=42
).values


# =========================================================
# SAVE BUNDLE
# =========================================================

bundle = {

    "model": final_classifier,

    "scaler": final_scaler,

    "background": background,

    "features": FEATURES,

    "task": "Baseline vs Stress",

    "label_mapping": {
        "0": "Baseline",
        "1": "Stress"
    }
}


joblib.dump(
    bundle,
    MODEL_PATH
)


# =========================================================
# SAVE METRICS
# =========================================================

metrics = {

    "accuracy": float(
        accuracy
    ),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1_score": float(
        f1
    ),

    "roc_auc": float(
        roc_auc
    ),

    "task": "Baseline vs Stress",

    "features": FEATURES,

    "evaluation": (
        "Subject-aware held-out test split"
    )
}


with open(
    METRICS_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print()
print("Model saved:")
print(MODEL_PATH)

print()
print("Metrics saved:")
print(METRICS_PATH)

print()
print("==========================================")
print("       TRAINING COMPLETED")
print("==========================================")