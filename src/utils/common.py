import json
import os
import tempfile
from functools import partial
from pathlib import Path

import joblib
import mlflow.sklearn
import optuna
import pandas as pd
import shap
import yaml
from box import ConfigBox
from dotenv import load_dotenv
from ensure import ensure_annotations
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import mlflow
from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.logger import logger


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    with open(path, "r", encoding="utf-8") as f:
        config = ConfigBox(yaml.safe_load(f))
        logger.info("yaml file: %s loaded successfully", path)
    return config


def prepare_train_val_data(train_path: str, val_path: str, target: str):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_val = val_df.drop(columns=[target])
    y_val = val_df[target]

    return X_train, y_train, X_val, y_val


def _log_shap_importance(final_pipeline, X_train, top_n: int = 10) -> dict:
    """
    Compute SHAP mean absolute values for top_n features.
    Logs each as an MLflow metric and saves a JSON artifact.
    Returns the top_features dict so callers can use it too.
    """
    try:
        preprocessor_fitted = final_pipeline.named_steps["preprocessor"]
        X_train_transformed = preprocessor_fitted.transform(X_train)
        inner_model = final_pipeline.named_steps["model"]

        explainer = shap.Explainer(inner_model, X_train_transformed)
        # Sample 200 rows for speed; increase for higher fidelity
        shap_values = explainer(X_train_transformed[:200])

        feature_names = preprocessor_fitted.get_feature_names_out()
        mean_abs_shap = abs(shap_values.values).mean(  # pylint: disable=no-member
            axis=0
        )
        top_indices = mean_abs_shap.argsort()[::-1][:top_n]
        top_features = {
            str(feature_names[i]): float(round(mean_abs_shap[i], 4))
            for i in top_indices
        }

        logger.info("Top %d SHAP features: %s", top_n, top_features)

        # Log each feature as an MLflow metric
        for feat, val in top_features.items():
            safe_key = ("shap_" + feat.replace(" ", "_"))[:50]
            mlflow.log_metric(safe_key, val)

        # Save as JSON artifact (FastAPI reads this at inference time)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(top_features, tmp)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="shap")
        os.unlink(tmp_path)

        return top_features

    except Exception as exc:
        logger.warning("SHAP logging skipped: %s", exc)
        return {}


def train_model(experiment_name: str, objective_func, model_type: str):
    logger.info("Starting Optuna Tuning for %s", model_type)
    load_dotenv()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    config = read_yaml(Path("src/config/config.yaml"))
    train_df = pd.read_csv(config.data_paths.train_data)
    val_df = pd.read_csv(config.data_paths.val_data)

    target_column = "Churn"
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    objective_func = partial(
        objective_func,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_df=train_df,
        target_column=target_column,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=30)
    logger.info("Best params: %s", study.best_params)

    # ── Final model training ─────────────────────────────────────────────────
    best_params = study.best_params
    best_params["random_state"] = 42

    final_preprocessor = build_preprocessing_pipeline(train_df, target_column)

    if model_type == "randomforest":
        model = RandomForestClassifier(class_weight="balanced", **best_params)
        registered_name = "rf-optuna-model"
    elif model_type == "logistig regression":
        # Note: typo preserved intentionally to match original codebase
        model = LogisticRegression(class_weight="balanced", **best_params)
        registered_name = "logreg-optuna-model"
    elif model_type == "xgboost":
        # XGBoost uses scale_pos_weight (already tuned in objective)
        model = XGBClassifier(**best_params)
        registered_name = "xgb-optuna-model"
    else:
        raise ValueError(f"Model type '{model_type}' not supported")

    final_pipeline = Pipeline(
        [
            ("preprocessor", final_preprocessor),
            ("model", model),
        ]
    )

    final_pipeline.fit(X_train, y_train)
    val_preds = final_pipeline.predict(X_val)

    acc = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)
    roc_auc = roc_auc_score(y_val, val_preds)

    logger.info("Final metrics — F1: %.4f, ROC AUC: %.4f", f1, roc_auc)

    with mlflow.start_run():
        mlflow.log_params(best_params)
        mlflow.log_metrics(
            {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
            }
        )

        # ── SHAP: log top-10 feature importances ────────────────────────────
        _log_shap_importance(final_pipeline, X_train, top_n=10)

        mlflow.sklearn.log_model(
            final_pipeline,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        # Save local backup model for Render deployment
        os.makedirs("models", exist_ok=True)
        joblib.dump(final_pipeline, "models/model.pkl")
        logger.info("Local model saved at models/model.pkl")

    mlflow.end_run()
