import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.pipelines.data_preprocessing import build_preprocessing_pipeline
from src.utils.common import train_model
from src.utils.logger import logger


def objective(trial, X_train, y_train, X_val, y_val, train_df, target_column):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_int("units_per_layer", 32, 256)
    hidden_layer_sizes = tuple([units] * n_layers) 

    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init", 1e-4, 1e-2, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 15,
        "random_state": 42,
    }

    preprocessor = build_preprocessing_pipeline(train_df, target_column)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", MLPClassifier(**params)),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds)

    with mlflow.start_run(nested=True):
        log_params = {k: v for k, v in params.items() if k != "hidden_layer_sizes"}
        log_params["hidden_layer_sizes"] = str(hidden_layer_sizes)
        mlflow.log_params(log_params)
        mlflow.log_metric("f1", f1)

    return f1


if __name__ == "__main__":
    logger.info("Train MLP (Neural Network) optuna")
    train_model("optuna-mlp", objective, "mlp")