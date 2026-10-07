from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes.prediction import router as prediction_router
from backend.routes.explanation import router as explanation_router
from backend.routes.federated import router as federated_router


app = FastAPI(
    title="PE-FedHealth",
    description=(
        "Privacy-Preserving and Explainable "
        "Federated Learning Framework"
    ),
    version="1.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "message": "PE-FedHealth API is running",
        "status": "Online",
        "docs": "/docs"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "PE-FedHealth FastAPI Backend"
    }


# =========================================================
# ROUTERS
# =========================================================

app.include_router(prediction_router)
app.include_router(explanation_router)
app.include_router(federated_router)


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():

    print()
    print("==========================================")
    print("       PE-FedHealth Backend Started")
    print("==========================================")
    print("FastAPI: http://127.0.0.1:8001")
    print("Swagger: http://127.0.0.1:8001/docs")
    print("==========================================")