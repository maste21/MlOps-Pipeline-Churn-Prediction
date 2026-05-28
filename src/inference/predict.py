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


def load_model():
    try:
        logger.info("Trying to load model from MLflow registry...")
        load_dotenv()
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        config = read_yaml(Path("src/config/config.yaml"))
        client = MlflowClient()
        latest_version = client.get_latest_versions(config.model_registry.name)[
            0
        ].version
        model_uri = f"models:/{config.model_registry.name}/{latest_version}"
        logger.info("Trying to load model from MLflow: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("MLflow model loaded successfully")
        return model
    except Exception as e:
        logger.warning("MLflow loading failed: %s", e)
        local_model_path = "models/model.pkl"
        logger.info("Loading local fallback model: %s", local_model_path)
        model = joblib.load(local_model_path)
        logger.info("Local fallback model loaded successfully")
        return model


def predict(input_data: dict):
    model = load_model()
    df = pd.DataFrame([input_data])
    probs = model.predict_proba(df)[0]
    no_prob = float(round(probs[0], 4))
    yes_prob = float(round(probs[1], 4))
    prediction = 1 if yes_prob >= 0.6 else 0
    return prediction, yes_prob, no_prob
