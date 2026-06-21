import json
import os
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

import mlflow
from src.pipelines.ensemble_model import ManualSoftVotingEnsemble
from src.utils.common import read_yaml
from src.utils.logger import logger

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
LOCAL_MODEL_PATH = CURRENT_DIR / "models" / "model.pkl"
LOCAL_SHAP_PATH = CURRENT_DIR / "models" / "shap_importance.json"

GLOBAL_MODEL = None
GLOBAL_MODEL_VERSION = "not_loaded_yet"
GLOBAL_SHAP_FEATURES: dict = {}
_FALLBACK_MODEL_NAMES = [
    "ensemble-voting-model",
    "lgbm-optuna-model",
    "xgb-optuna-model",
    "rf-optuna-model",
    "mlp-optuna-model",
    "logreg-optuna-model",
]


def load_model():
    global GLOBAL_MODEL_VERSION

    try:
        logger.info("Trying to load model from MLflow registry...")
        load_dotenv()

        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI environment variable is not set.")

        mlflow.set_tracking_uri(tracking_uri)
        config = read_yaml(Path("src/config/config.yaml"))
        client = MlflowClient()

        primary_name = config.model_registry.name
        model_names_to_try = [primary_name] + [
            m for m in _FALLBACK_MODEL_NAMES if m != primary_name
        ]

        for model_name in model_names_to_try:
            try:
                versions = client.get_latest_versions(model_name)
                if not versions:
                    logger.info("'%s' not registered yet — trying next.", model_name)
                    continue

                latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
                model_uri = f"models:/{model_name}/{latest.version}"
                logger.info("Loading model from MLflow: %s", model_uri)
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("MLflow model loaded successfully: %s", model_name)
                GLOBAL_MODEL_VERSION = f"mlflow://{model_name}/v{latest.version}"
                _load_shap_features()
                return model

            except Exception as inner_exc:
                logger.warning(
                    "Could not load '%s': %s — trying next.", model_name, inner_exc
                )
                continue

        raise RuntimeError("No model found in MLflow registry after trying all names.")

    except Exception as exc:
        logger.warning("MLflow loading failed: %s", exc)
        logger.info("Loading local fallback model from: %s", LOCAL_MODEL_PATH)

        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Critical Error: Fallback model not found at {LOCAL_MODEL_PATH}. "
                "Run training pipeline first."
            )

        model = joblib.load(LOCAL_MODEL_PATH)
        logger.info("Local fallback model loaded successfully.")
        GLOBAL_MODEL_VERSION = "local://models/model.pkl"

    _load_shap_features()
    return model


def _load_shap_features():
    """Populate GLOBAL_SHAP_FEATURES from local JSON or MLflow artifact."""
    global GLOBAL_SHAP_FEATURES  # noqa: PLW0603

    try:
        if LOCAL_SHAP_PATH.exists():
            with open(LOCAL_SHAP_PATH, "r", encoding="utf-8") as f:
                GLOBAL_SHAP_FEATURES = json.load(f)
            logger.info("SHAP features loaded from %s", LOCAL_SHAP_PATH)
            return
    except Exception as exc:
        logger.warning("Local SHAP load failed: %s", exc)

    try:
        client = MlflowClient()
        config = read_yaml(Path("src/config/config.yaml"))
        model_name = config.model_registry.name
        versions = client.get_latest_versions(model_name)
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            run_id = latest.run_id
            artifacts = client.list_artifacts(run_id, path="shap")
            shap_artifact = next(
                (a for a in artifacts if a.path.endswith(".json")), None
            )
            if shap_artifact:
                local_path = client.download_artifacts(run_id, shap_artifact.path)
                with open(local_path, "r", encoding="utf-8") as f:
                    GLOBAL_SHAP_FEATURES = json.load(f)
                logger.info(
                    "SHAP features loaded from MLflow artifact: %s",
                    shap_artifact.path,
                )
                return
    except Exception as exc:
        logger.warning("MLflow SHAP artifact load failed: %s", exc)

    logger.warning("SHAP features unavailable — top_features will be empty.")
    GLOBAL_SHAP_FEATURES = {}


def predict(input_data: dict):
    global GLOBAL_MODEL

    if GLOBAL_MODEL is None:
        logger.info("Model uninitialized for this worker. Executing load sequence...")
        GLOBAL_MODEL = load_model()

    df = pd.DataFrame([input_data])

    if hasattr(GLOBAL_MODEL, "predict_proba"):
        probs = GLOBAL_MODEL.predict_proba(df)[0]
    elif hasattr(GLOBAL_MODEL, "_model_impl") and hasattr(
        GLOBAL_MODEL._model_impl, "predict_proba"
    ):
        probs = GLOBAL_MODEL._model_impl.predict_proba(df)[0]
    else:
        raise AttributeError(
            "The loaded model variant does not expose 'predict_proba'."
        )

    no_prob = float(round(probs[0], 4))
    yes_prob = float(round(probs[1], 4))
    prediction = 1 if yes_prob >= 0.6 else 0

    return prediction, yes_prob, no_prob
