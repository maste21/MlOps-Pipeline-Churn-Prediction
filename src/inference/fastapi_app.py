"""
FastAPI REST endpoint for churn prediction.
Endpoints:
  GET  /health   → status + model version (always 200, never crashes)
  GET  /reload   → manually trigger model reload
  POST /predict  → churn label, probability, top-10 SHAP feature importances
"""

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

try:
    from pydantic import BaseModel
except ImportError:
    from pydantic.v1 import BaseModel

import src.inference.predict as predict_engine
from src.utils.logger import logger

app = FastAPI(
    title="Churn Prediction API",
    description="REST API for customer churn prediction with SHAP explainability.",
    version="1.0.0",
)

load_dotenv()


# ── Pydantic schema ───────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "1234-XYZ",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 820.50,
            }
        }


# ── Global SHAP Local Fallback Parser ──────────────────────────────────────────
def _load_local_shap_features() -> dict:
    """Attempts to read local SHAP artifacts if they exist in the workspace."""
    try:
        shap_path = predict_engine.REPO_ROOT / "models" / "shap_importance.json"
        if shap_path.exists():
            with open(shap_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.info("Local SHAP parameter ingestion skipped: %s", e)
    return {}


@app.on_event("startup")
def startup_event():
    logger.info("FastAPI starting — Model registry router mapped to lazy evaluator.")
    try:
        predict_engine.load_model()
    except Exception as e:
        logger.warning("Pre-warm startup loading sequence bypassed: %s", e)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_version": predict_engine.GLOBAL_MODEL_VERSION}


@app.get("/reload")
def reload_model():
    """Manually trigger model reload — handles both MLflow and local fallback context safely."""
    try:
        predict_engine.GLOBAL_MODEL = predict_engine.load_model()
        return {
            "status": "reloaded",
            "model_version": predict_engine.GLOBAL_MODEL_VERSION,
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Reload process failed: {str(e)}",
        )


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    try:
        input_dict = customer.dict()
        prediction, yes_prob, no_prob = predict_engine.predict(input_dict)
        shap_features = _load_local_shap_features()

        return {
            "churn": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "probability": yes_prob,
            "no_churn_probability": no_prob,
            "top_features": shap_features,
            "model_version": predict_engine.GLOBAL_MODEL_VERSION,
        }
    except Exception as exc:
        logger.error("Prediction route processing failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.inference.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
