from unittest.mock import MagicMock, patch

from box import ConfigBox

from src.inference.predict import load_model


@patch("src.inference.predict.read_yaml")
@patch("mlflow.sklearn.load_model")
@patch("src.inference.predict.MlflowClient")
def test_load_model(mock_mlflow_client, mock_sklearn_load, mock_read_yaml):
    mock_read_yaml.return_value = ConfigBox(
        {"model_registry": {"name": "test_model", "stage": "Production"}}
    )

    mock_version_obj = MagicMock()
    mock_version_obj.version = "1"

    mock_client_instance = MagicMock()
    mock_client_instance.get_latest_versions.return_value = [mock_version_obj]
    mock_mlflow_client.return_value = mock_client_instance

    expected_model = MagicMock()
    mock_sklearn_load.return_value = expected_model
    model = load_model()

    mock_sklearn_load.assert_any_call("models:/test_model/1")
    assert model == expected_model
