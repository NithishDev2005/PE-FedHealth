import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# -----------------------------------------
# Paths
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "final_features.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

SCALER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# -----------------------------------------
# Load data
# -----------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["subject", "label"])
y = df["label"]


# -----------------------------------------
# Same train-test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# Load model and scaler
# -----------------------------------------

model = joblib.load(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)


# -----------------------------------------
# Scale test data
# -----------------------------------------

X_test_scaled = scaler.transform(X_test)


# -----------------------------------------
# Predictions
# -----------------------------------------

y_pred = model.predict(X_test_scaled)


# -----------------------------------------
# Metrics
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("\n========== MODEL EVALUATION ==========")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# -----------------------------------------
# Classification report
# -----------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------------------
# Confusion matrix
# -----------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")

print(cm)


# -----------------------------------------
# Save confusion matrix
# -----------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path)

plt.show()


# -----------------------------------------
# Save metrics
# -----------------------------------------

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics_path = os.path.join(
    RESULTS_DIR,
    "metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


print("\nEvaluation completed!")

print("Metrics saved to:")
print(metrics_path)

print("\nConfusion matrix saved to:")
print(cm_path)