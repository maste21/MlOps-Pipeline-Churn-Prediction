import mlflow.sklearn
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

import mlflow
from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import train_model
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val, train_df, target_column):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "is_unbalance": True,
        "random_state": 42,
        "verbose": -1,  # suppress noisy training output
    }

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**params)),
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
    logger.info("Train LightGBM optuna")
    train_model("optuna-lightgbm", objective, "lightgbm")
