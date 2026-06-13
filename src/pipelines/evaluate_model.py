import os
from pathlib import Path

import mlflow.sklearn
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
from src.utils.common import read_yaml
from src.utils.logger import logger

MODEL_NAMES = [
    "ensemble-voting-model",
    "lgbm-optuna-model",
    "xgb-optuna-model",
    "rf-optuna-model",
    "mlp-optuna-model",
    "logreg-optuna-model",
]


def _load_model_safe(client: MlflowClient, model_name: str):
    """Load latest version of a model. Returns (model, version_str) or (None, None)."""
    try:
        versions = client.get_latest_versions(model_name)
        if not versions:
            logger.warning("Model '%s' not found in registry — skipping.", model_name)
            return None, None
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        model_uri = f"models:/{model_name}/{latest.version}"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded '%s' v%s for evaluation.", model_name, latest.version)
        return model, latest.version
    except Exception as exc:
        logger.warning("Could not load '%s': %s — skipping.", model_name, exc)
        return None, None


def evaluate_model(config_path: Path):
    logger.info("Starting model evaluation")
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("model-evaluation")

    config = read_yaml(config_path)
    test_path = Path(config.data_paths.test_data)

    df = pd.read_csv(test_path)
    logger.info("Loaded test set with shape %s", df.shape)

    target_column = "Churn"
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    client = MlflowClient()

    primary_name = config.model_registry.name
    primary_model, primary_version = _load_model_safe(client, primary_name)

    if primary_model is None:
        logger.error(
            "Primary model '%s' not found. " "Run training pipeline first (make all).",
            primary_name,
        )
        return

    y_pred = primary_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    logger.info(
        "Primary model '%s' v%s — Accuracy: %.4f | F1: %.4f | ROC-AUC: %.4f",
        primary_name,
        primary_version,
        acc,
        f1,
        roc_auc,
    )

    with mlflow.start_run(run_name=f"eval-{primary_name}"):
        mlflow.log_param("model_name", primary_name)
        mlflow.log_param("model_version", primary_version)
        mlflow.log_metrics(
            {
                "test_accuracy": acc,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1,
                "test_roc_auc": roc_auc,
            }
        )

    logger.info("Evaluating all registered models for comparison...")
    comparison_metrics = {}

    for model_name in MODEL_NAMES:
        model, version = _load_model_safe(client, model_name)
        if model is None:
            continue
        try:
            preds = model.predict(X_test)
            metrics = {
                "accuracy": round(accuracy_score(y_test, preds), 4),
                "f1": round(f1_score(y_test, preds), 4),
                "roc_auc": round(roc_auc_score(y_test, preds), 4),
                "precision": round(precision_score(y_test, preds), 4),
                "recall": round(recall_score(y_test, preds), 4),
            }
            comparison_metrics[model_name] = metrics
            logger.info(
                "  %-30s v%-3s F1=%.4f  AUC=%.4f",
                model_name,
                version,
                metrics["f1"],
                metrics["roc_auc"],
            )
        except Exception as exc:
            logger.warning("Evaluation failed for '%s': %s", model_name, exc)

    if comparison_metrics:
        with mlflow.start_run(run_name="all-models-comparison"):
            for model_name, metrics in comparison_metrics.items():
                short = model_name.replace("-optuna-model", "").replace("-", "_")
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{short}_{metric_name}", value)

        logger.info("Model comparison logged to MLflow experiment 'model-evaluation'.")


if __name__ == "__main__":
    evaluate_model(Path("src/config/config.yaml"))
