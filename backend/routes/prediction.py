from fastapi import APIRouter
from pydantic import BaseModel

from ml.disease_risk import calculate_risk


router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"]
)


class SensorData(BaseModel):
    ecg_std: float
    eda_mean: float
    temp_std: float
    resp_std: float


@router.post("/risk")
def predict_risk(data: SensorData):

    features = {
        "ecg_std": data.ecg_std,
        "eda_mean": data.eda_mean,
        "temp_std": data.temp_std,
        "resp_std": data.resp_std
    }

    risk = calculate_risk(features)

    return {
        "message": (
            "ML-based physiological risk "
            "prediction generated successfully"
        ),
        "prediction": risk
    }