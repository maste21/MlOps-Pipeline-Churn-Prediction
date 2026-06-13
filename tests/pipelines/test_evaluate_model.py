from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.pipelines.evaluate_model import evaluate_model


@patch("src.pipelines.evaluate_model.mlflow")
@patch("src.pipelines.evaluate_model.read_yaml")
@patch("src.pipelines.evaluate_model.MlflowClient")
def test_evaluate_model(mock_mlflow_client, mock_read_yaml, mock_mlflow, tmp_path):
    model_registry_mock = MagicMock()
    model_registry_mock.name = "test_model"
    model_registry_mock.stage = "Staging"

    mock_read_yaml.return_value = MagicMock(
        data_paths=MagicMock(test_data=str(tmp_path / "test.csv")),
        model_registry=model_registry_mock,
    )

    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4],
            "feature2": [10, 20, 30, 40],
            "Churn": [0, 1, 0, 1],
        }
    )
    df.to_csv(tmp_path / "test.csv", index=False)

    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1, 0, 1]
    mock_mlflow.sklearn.load_model.return_value = mock_model

    mock_client_instance = MagicMock()
    mock_client_instance.get_latest_versions.return_value = [MagicMock(version="1")]
    mock_mlflow_client.return_value = mock_client_instance

    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    evaluate_model(Path("fake_config_path.yaml"))

    assert mock_model.predict.called, "The model's predict method was never called."
    assert (
        mock_model.predict.call_count > 0
    ), "Expected predict to be executed at least once."

    assert mock_mlflow.log_metrics.called, "MLflow metrics were never logged."
    assert (
        mock_mlflow.log_metrics.call_count > 0
    ), "Expected log_metrics to be executed at least once."
