import os
import subprocess

from prefect import flow, task


def get_env_with_mlflow():
    """Helper to forward the system environment variables along with MLflow configuration."""
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050")
    return env


@task(retries=3, retry_delay_seconds=2)
def ingest_data():
    subprocess.run(["python", "-m", "src.pipelines.data_ingestion"], check=True)


@task
def split_data():
    subprocess.run(["python", "-m", "src.pipelines.data_split"], check=True)


@task(log_prints=True)
def train_random_forest():
    subprocess.run(
        ["python", "-m", "src.pipelines.train_optuna_rf"],
        env=get_env_with_mlflow(),
        check=True,
    )


@task(log_prints=True)
def train_xgboost():
    subprocess.run(
        ["python", "-m", "src.pipelines.train_optuna_xgb"],
        env=get_env_with_mlflow(),
        check=True,
    )


@task(log_prints=True)
def train_lightgbm():
    """LightGBM — fastest gradient boosting, best for telecom churn benchmarks."""
    subprocess.run(
        ["python", "-m", "src.pipelines.train_optuna_lgbm"],
        env=get_env_with_mlflow(),
        check=True,
    )


@task(log_prints=True)
def train_mlp():
    """MLP Neural Network — sklearn, no extra dependencies, pure tabular NN."""
    subprocess.run(
        ["python", "-m", "src.pipelines.train_optuna_mlp"],
        env=get_env_with_mlflow(),
        check=True,
    )


@task(log_prints=True)
def train_ensemble():
    """Soft Voting Ensemble — combines all trained models for best accuracy."""
    subprocess.run(
        ["python", "-m", "src.pipelines.train_ensemble"],
        env=get_env_with_mlflow(),
        check=True,
    )


@task(log_prints=True)
def evaluate_model():
    subprocess.run(
        ["python", "-m", "src.pipelines.evaluate_model"],
        env=get_env_with_mlflow(),
        check=True,
    )


@flow(name="Churn Prediction Training Pipeline")
def training_pipeline():
    ingest_data()
    split_data()

    # All 5 base models run in parallel
    rf   = train_random_forest.submit()
    xgb  = train_xgboost.submit()
    lgbm = train_lightgbm.submit()
    mlp  = train_mlp.submit()

    # Wait for all base models before building ensemble
    rf.result()
    xgb.result()
    lgbm.result()
    mlp.result()

    # Ensemble after all base models are registered
    train_ensemble()

    evaluate_model()


if __name__ == "__main__":
    training_pipeline()