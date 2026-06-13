import os
from pathlib import Path

import joblib
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import mlflow
from src.pipelines.ensemble_model import ManualSoftVotingEnsemble  # noqa: F401
from src.utils.common import read_yaml
from src.utils.logger import logger


def load_model_from_registry(client: MlflowClient, model_name: str):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.warning("Model '%s' not found in registry — skipping.", model_name)
            return None, None
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{model_name}/{latest.version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded %s v%s", model_name, latest.version)
        return model, latest.version
    except Exception as exc:
        logger.warning("Could not load %s: %s — skipping.", model_name, exc)
        return None, None


# ── Main ──────────
def train_ensemble():
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("optuna-ensemble")

    config = read_yaml(Path("src/config/config.yaml"))
    val_df = pd.read_csv(config.data_paths.val_data)
    test_df = pd.read_csv(config.data_paths.test_data)

    target_column = "Churn"
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]

    client = MlflowClient()

    # Load all base models from MLflow registry
    base_models = []
    for name, registry_name in [
        ("logreg", "logreg-optuna-model"),
        ("rf", "rf-optuna-model"),
        ("xgb", "xgb-optuna-model"),
        ("lgbm", "lgbm-optuna-model"),
        ("mlp", "mlp-optuna-model"),
    ]:
        model, version = load_model_from_registry(client, registry_name)
        if model is not None:
            base_models.append((name, model))

    if len(base_models) < 2:
        logger.error(
            "Need at least 2 base models for ensemble. Found: %d. "
            "Run training pipeline first (make train_all).",
            len(base_models),
        )
        return

    logger.info(
        "Building ManualSoftVotingEnsemble with %d models: %s",
        len(base_models),
        [n for n, _ in base_models],
    )

    ensemble = ManualSoftVotingEnsemble(estimators=base_models)
    ensemble.fit(X_val, y_val)

    test_preds = ensemble.predict(X_test)
    test_proba = ensemble.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)
    roc_auc = roc_auc_score(y_test, test_proba)

    logger.info(
        "Ensemble — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | "
        "F1: %.4f | ROC-AUC: %.4f",
        acc,
        precision,
        recall,
        f1,
        roc_auc,
    )

    with mlflow.start_run(run_name="soft-voting-ensemble"):
        mlflow.log_param("base_models", str([n for n, _ in base_models]))
        mlflow.log_param("voting", "soft_manual_average")
        mlflow.log_param("n_base_models", len(base_models))
        mlflow.log_metrics(
            {
                "test_accuracy": acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
            }
        )

        mlflow.sklearn.log_model(
            ensemble,
            artifact_path="model",
            registered_model_name="ensemble-voting-model",
        )
        logger.info(
            "Ensemble registered as 'ensemble-voting-model' | ROC-AUC=%.4f",
            roc_auc,
        )

        os.makedirs("models", exist_ok=True)
        joblib.dump(ensemble, "models/model.pkl")
        logger.info("Latest ensemble model saved locally at models/model.pkl")

    mlflow.end_run()


if __name__ == "__main__":
    logger.info("Building Soft Voting Ensemble from all registered models")
    train_ensemble()
