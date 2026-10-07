from fastapi import APIRouter

from federated.server import run_federated_training


router = APIRouter(
    prefix="/federated",
    tags=["Federated Learning"]
)


# =========================================================
# STATUS
# =========================================================

@router.get("/status")
def federated_status():

    return {
        "status": (
            "Federated Learning system "
            "is running"
        ),
        "clients": 15,
        "privacy": "Enabled",
        "aggregation": "Weighted Aggregation"
    }


# =========================================================
# TRAINING
# =========================================================

@router.post("/train")
def federated_train():

    result = run_federated_training()

    return result