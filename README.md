# PE-FedHealth

## Privacy-Preserving and Explainable Federated Learning Framework for Multi-Disease Risk Prediction Using Wearable Sensor Data

PE-FedHealth is a research-oriented federated learning framework designed to explore privacy-aware and explainable machine learning using wearable sensor data.

The system combines Federated Learning, noise-based privacy protection, weighted model aggregation, and SHAP-based explainability with a web-based interface.

> **Important:** The current WESAD-based implementation evaluates **Baseline vs Stress physiological states**. WESAD is a wearable stress and affect dataset, and its labels should not be interpreted as clinical disease diagnoses. Therefore, the current outputs are research-oriented physiological state/risk indicators rather than medical diagnoses.

---

## Key Features

- Wearable sensor data processing
- Baseline vs Stress state prediction
- Federated Learning with multiple clients
- 15 simulated federated clients
- 5 federated training rounds
- Weighted model aggregation
- Noise-based Differential Privacy mechanism
- SHAP-based model explainability
- FastAPI backend
- React + Vite frontend
- Interactive prediction dashboard
- Privacy monitoring interface
- Federated training reports

---

## Dataset

### WESAD

The project uses the **WESAD (Wearable Stress and Affect Detection)** dataset.

The current implementation uses wearable physiological features including:

- ECG standard deviation
- EDA mean
- Temperature standard deviation
- Respiratory standard deviation

The implemented machine learning task is:

**Baseline vs Stress**

The WESAD labels are not clinical disease diagnoses.

---

## System Architecture

```text
                Wearable Sensor Data
                        |
                        v
                Feature Extraction
                        |
                        v
              Local Client Datasets
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Client 1      Client 2      Client 15
          |             |             |
          +-------------+-------------+
                        |
                        v
              Local Model Training
                        |
                        v
             Privacy Protection
              Noise-based Mechanism
                        |
                        v
              Weighted Aggregation
                        |
                        v
                Global Model
                        |
             +----------+----------+
             |                     |
             v                     v
       Risk Prediction       SHAP Explanation
             |                     |
             +----------+----------+
                        |
                        v
                 FastAPI Backend
                        |
                        v
                 React + Vite UI