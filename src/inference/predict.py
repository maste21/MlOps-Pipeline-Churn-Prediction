import os
from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

import mlflow
from src.utils.common import read_yaml
from src.utils.logger import logger

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
LOCAL_MODEL_PATH = REPO_ROOT / "models" / "model.pkl"

GLOBAL_MODEL = None
GLOBAL_MODEL_VERSION = "not_loaded_yet"


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

        latest_version = client.get_latest_versions(config.model_registry.name)[
            0
        ].version
        model_uri = f"models:/{config.model_registry.name}/{latest_version}"

        logger.info("Trying to load model from MLflow: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("MLflow model loaded successfully")
        GLOBAL_MODEL_VERSION = (
            f"mlflow://{config.model_registry.name}/v{latest_version}"
        )
        return model

    except Exception as e:
        logger.warning("MLflow loading failed: %s", e)
        logger.info(
            "Loading local fallback model from absolute path: %s", LOCAL_MODEL_PATH
        )

        if not LOCAL_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Critical Error: Fallback model not found at {LOCAL_MODEL_PATH}"
            )

        model = joblib.load(LOCAL_MODEL_PATH)
        logger.info("Local fallback model loaded successfully")
        GLOBAL_MODEL_VERSION = "local://models/model.pkl"
        return model


def predict(input_data: dict):
    """Uses Lazy Loading to ensure the model is safely initialized in memory across threads."""
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
