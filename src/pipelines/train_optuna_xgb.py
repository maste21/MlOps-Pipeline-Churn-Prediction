import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import mlflow
from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import train_model
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val, train_df, target_column):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    default_spw = round(neg / pos, 2)

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", default_spw * 0.5, default_spw * 2.0
        ),
        "random_state": 42,
        "eval_metric": "logloss",
    }

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds)

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("f1", f1)

    return f1


if __name__ == "__main__":
    logger.info("Train XGBoost optuna")
    train_model("optuna-xgboost", objective, "xgboost")
