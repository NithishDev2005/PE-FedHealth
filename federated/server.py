from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

from federated.client import FederatedClient
from privacy.secure_aggregation import secure_average


# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "final_features.csv"
)


# =========================================================
# CONFIGURATION
# =========================================================

FEATURES = [
    "ecg_std",
    "eda_mean",
    "temp_std",
    "resp_std"
]

NUMBER_OF_ROUNDS = 5


# =========================================================
# FEDERATED TRAINING
# =========================================================

def run_federated_training():

    print()
    print("==========================================")
    print("     FEDERATED TRAINING STARTED")
    print("==========================================")


    # =====================================================
    # LOAD DATA
    # =====================================================

    df = pd.read_csv(
        DATA_PATH
    )


    # =====================================================
    # BASELINE VS STRESS
    # =====================================================

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


    df = df.dropna(
        subset=FEATURES + [
            "subject",
            "target"
        ]
    )


    # =====================================================
    # CREATE CLIENTS
    # =====================================================

    client_data = {}


    for subject, group in df.groupby(
        "subject"
    ):

        X_subject = (
            group[FEATURES]
            .values
        )

        y_subject = (
            group["target"]
            .values
        )


        if len(
            np.unique(y_subject)
        ) < 2:

            continue


        client_data[
            str(subject)
        ] = (
            X_subject,
            y_subject
        )


    print(
        "Number of clients:",
        len(client_data)
    )


    # =====================================================
    # TRAIN / TEST SPLIT PER CLIENT
    # =====================================================

    train_clients = {}

    test_clients = {}


    for (
        client_id,
        (
            X_client,
            y_client
        )
    ) in client_data.items():

        try:

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = train_test_split(
                X_client,
                y_client,
                test_size=0.20,
                random_state=42,
                stratify=y_client
            )

        except ValueError:

            (
                X_train,
                X_test,
                y_train,
                y_test
            ) = train_test_split(
                X_client,
                y_client,
                test_size=0.20,
                random_state=42
            )


        train_clients[
            client_id
        ] = (
            X_train,
            y_train
        )


        test_clients[
            client_id
        ] = (
            X_test,
            y_test
        )


    # =====================================================
    # GLOBAL SCALER
    # =====================================================

    all_training_features = np.vstack([
        value[0]
        for value in train_clients.values()
    ])


    scaler = StandardScaler()

    scaler.fit(
        all_training_features
    )


    # =====================================================
    # CREATE FEDERATED CLIENT OBJECTS
    # =====================================================

    clients = []


    for (
        client_id,
        (
            X_train,
            y_train
        )
    ) in train_clients.items():

        X_scaled = scaler.transform(
            X_train
        )


        client = FederatedClient(
            X_scaled,
            y_train,
            client_id
        )


        clients.append(
            client
        )


    classes = np.array([
        0,
        1
    ])


    # =====================================================
    # GLOBAL PARAMETERS
    # =====================================================

    global_parameters = None


    # =====================================================
    # TRAINING ROUNDS
    # =====================================================

    for round_number in range(
        1,
        NUMBER_OF_ROUNDS + 1
    ):

        print()
        print(
            f"Federated Round "
            f"{round_number}/"
            f"{NUMBER_OF_ROUNDS}"
        )


        client_updates = []


        # -------------------------------------------------
        # LOCAL TRAINING
        # -------------------------------------------------

        for client in clients:

            try:

                update = client.train(
                    global_parameters,
                    classes
                )

                client_updates.append(
                    update
                )

            except Exception as error:

                print(
                    f"Client "
                    f"{client.client_id} "
                    f"failed:",
                    error
                )


        # -------------------------------------------------
        # AGGREGATION
        # -------------------------------------------------

        if not client_updates:

            raise RuntimeError(
                "No clients completed training."
            )


        global_parameters = (
            secure_average(
                client_updates
            )
        )


        print(
            "Global model updated."
        )


    # =====================================================
    # HELD-OUT EVALUATION
    # =====================================================

    predictions = []

    actual = []

    decision_scores = []

    client_results = []


    # -----------------------------------------------------
    # CREATE GLOBAL MODEL
    # -----------------------------------------------------

    global_model = FederatedClient(
        np.zeros((1, len(FEATURES))),
        np.array([0]),
        "global"
    )


    global_model.model.coef_ = (
        global_parameters[0].copy()
    )

    global_model.model.intercept_ = (
        global_parameters[1].copy()
    )

    global_model.model.classes_ = (
        classes.copy()
    )


    # -----------------------------------------------------
    # TEST EACH CLIENT
    # -----------------------------------------------------

    for (
        client_id,
        (
            X_test,
            y_test
        )
    ) in test_clients.items():

        X_test_scaled = scaler.transform(
            X_test
        )


        prediction = (
            global_model.model.predict(
                X_test_scaled
            )
        )


        scores = (
            global_model.model.decision_function(
                X_test_scaled
            )
        )


        predictions.extend(
            prediction.tolist()
        )

        actual.extend(
            y_test.tolist()
        )

        decision_scores.extend(
            scores.tolist()
        )


        client_accuracy = (
            accuracy_score(
                y_test,
                prediction
            )
        )


        client_results.append({
            "client_id": client_id,
            "samples": int(len(y_test)),
            "accuracy": round(
                float(client_accuracy),
                4
            )
        })


    # =====================================================
    # GLOBAL METRICS
    # =====================================================

    global_accuracy = accuracy_score(
        actual,
        predictions
    )


    global_precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )


    global_recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )


    global_f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )


    try:

        global_roc_auc = roc_auc_score(
            actual,
            decision_scores
        )

    except ValueError:

        global_roc_auc = 0.0


    print()
    print("==========================================")
    print("     FEDERATED TRAINING COMPLETED")
    print("==========================================")

    print(
        "Global held-out test accuracy:",
        round(
            float(global_accuracy),
            4
        )
    )

    print(
        "Total held-out test samples:",
        len(actual)
    )


    # =====================================================
    # RETURN API RESULT
    # =====================================================

    return {

        "message":
            "Federated training completed successfully.",

        "clients_participated":
            len(clients),

        "rounds":
            NUMBER_OF_ROUNDS,

        "task":
            "Baseline vs Stress",

        "label_mapping": {
            "0": "Baseline",
            "1": "Stress"
        },

        "privacy":
            "Differential Privacy Enabled",

        "aggregation":
            "Weighted Aggregation",

        "evaluation":
            "Held-out Baseline/Stress test data",

        "global_model": {

            "status":
                "Updated",

            "accuracy":
                round(
                    float(global_accuracy),
                    4
                ),

            "precision":
                round(
                    float(global_precision),
                    4
                ),

            "recall":
                round(
                    float(global_recall),
                    4
                ),

            "f1_score":
                round(
                    float(global_f1),
                    4
                ),

            "roc_auc":
                round(
                    float(global_roc_auc),
                    4
                )
        },

        "client_results":
            client_results
    }