"""
FastAPI REST endpoint for churn prediction.
Endpoints:
  GET  /health   → status + model version (always 200, never crashes)
  GET  /reload   → manually trigger model reload from MLflow
  POST /predict  → churn label, probability, top-10 SHAP feature importances
"""

import json
import os
import threading
import time
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel  # pylint: disable=no-name-in-module

import mlflow
from src.utils.common import read_yaml
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
        schema_extra = {
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


# ── Global model state ────────────────────────────────────────────────────────
_model = None
_model_version = "not_loaded_yet"
_shap_features: dict = {}
_model_names = ["xgb-optuna-model", "rf-optuna-model", "logreg-optuna-model"]


def _try_load_once() -> bool:
    """
    Attempts to load the best available registered model version dynamically.
    Uses native PyFunc URI resolution to bypass underlying file system tracking bugs.
    """
    global _model, _model_version, _shap_features

    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        # ── Step 1: Discover registered models ────────────────────────────
        try:
            all_models = client.search_registered_models()
            registered_names = {m.name for m in all_models}
            logger.info("Registered models found in MLflow: %s", registered_names)
        except Exception as e:
            logger.warning("Could not connect to MLflow Registry: %s", e)
            return False

        if not registered_names:
            logger.info("MLflow Registry is currently empty.")
            return False

        # ── Step 2: Choose target model by project preference order ───────
        chosen_name = None
        for name in _model_names:
            if name in registered_names:
                chosen_name = name
                break

        if chosen_name is None:
            chosen_name = next(iter(registered_names))
            logger.info("Falling back to available model: %s", chosen_name)

        # ── Step 3: Extract the absolute latest version entry ────────────
        try:
            versions = client.search_model_versions(f"name='{chosen_name}'")
            if not versions:
                logger.warning("No versions found for model sequence: %s", chosen_name)
                return False

            latest_version_meta = sorted(
                versions, key=lambda v: int(v.version), reverse=True
            )[0]
            v_num = latest_version_meta.version
            run_id = latest_version_meta.run_id
        except Exception as e:
            logger.warning("Failed parsing version schema for %s: %s", chosen_name, e)
            return False

        # ── Step 4: Stream Model Engine Live via PyFunc ───────────────────
        _model_version = f"{chosen_name}/v{v_num}"
        model_uri = f"models:/{chosen_name}/{v_num}"
        logger.info(
            "Targeting model entry: %s via Registry URI: %s", _model_version, model_uri
        )

        try:
            logger.info("Streaming model binaries over HTTP via PyFunc Registry API...")
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Model compiled cleanly into container memory layer.")
        except Exception as load_err:
            logger.warning("PyFunc network resolution strategy blocked: %s", load_err)
            return False

        # ── Step 5: Read SHAP feature importances (Optional) ──────────────
        shap_data = {}
        try:
            logger.info("Downloading SHAP metrics payload remotely via Client App...")
            local_artifact_root = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="shap"
            )
            if os.path.exists(local_artifact_root):
                json_files = list(Path(local_artifact_root).glob("*.json"))
                if json_files:
                    with open(json_files[0], "r") as f:
                        shap_data = json.load(f)
                    logger.info("Successfully loaded SHAP feature priorities.")
        except Exception as shap_err:
            logger.info("SHAP parameter ingestion skipped (Non-critical): %s", shap_err)

        # ── Step 6: Commit state live to global parameters ───────────────────
        _model = loaded_model
        _shap_features = shap_data
        logger.info("Model pipeline state validated successfully: %s", _model_version)
        return True

    except Exception as outer_err:
        logger.warning(
            "Critical bottleneck hit during model generation cycle: %s", outer_err
        )
        return False


def _background_loader(retry_interval: int = 30, max_attempts: int = 40):
    """Retry every 30s for up to 20 min waiting for training to finish."""
    for attempt in range(1, max_attempts + 1):
        logger.info("Model load retry %d/%d...", attempt, max_attempts)
        if _try_load_once():
            logger.info("Background loader: model ready — %s", _model_version)
            return
        time.sleep(retry_interval)
    logger.error("Gave up after %d attempts. Call GET /reload manually.", max_attempts)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    logger.info("FastAPI starting — attempting model load...")
    if not _try_load_once():
        logger.warning("No model yet — background retry thread started (every 30s).")
        threading.Thread(target=_background_loader, daemon=True).start()


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    if _model is None:
        return {
            "status": "waiting",
            "detail": "Model not loaded yet. Training may still be running or paths are aligning. "
            "Call GET /reload once training finishes.",
            "model_version": _model_version,
        }
    return {"status": "ok", "model_version": _model_version}


@app.get("/reload")
def reload_model():
    """Manually trigger model reload — call this after training completes."""
    success = _try_load_once()
    if success:
        return {"status": "reloaded", "model_version": _model_version}
    raise HTTPException(
        status_code=503,
        detail=f"No registered model found or files not accessible. Status: {_model_version}",
    )


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Check GET /health for status.",
        )
    try:
        # customer.dict() now naturally preserves 'customerID' inside the DataFrame context
        input_df = pd.DataFrame([customer.dict()])

        # PyFunc models use the uniform .predict() architecture
        prediction_raw = _model.predict(input_df)

        # Safely capture predictions whether returned as a dataframe, list, or series
        if isinstance(prediction_raw, pd.DataFrame):
            prediction = int(prediction_raw.iloc[0, 0])
        elif hasattr(prediction_raw, "tolist"):
            prediction = int(prediction_raw[0])
        else:
            prediction = int(prediction_raw)

        # PyFunc wraps predict_proba under internal flavor logic.
        # If your underlying model returns probabilities inside standard outputs, we fetch it here:
        probability = 0.0
        if hasattr(_model, "predict_proba"):
            probability = float(round(_model.predict_proba(input_df)[0][1], 4))
        elif hasattr(_model._model_impl, "predict_proba"):
            probability = float(
                round(_model._model_impl.predict_proba(input_df)[0][1], 4)
            )

        return {
            "churn": prediction,
            "churn_label": "Churn" if prediction == 1 else "No Churn",
            "probability": probability,
            "top_features": _shap_features,
            "model_version": _model_version,
        }
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.inference.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
