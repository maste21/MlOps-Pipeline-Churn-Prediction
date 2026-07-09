from unittest.mock import MagicMock, patch

import pytest
from box import ConfigBox

from src.inference.predict import load_model, predict


@patch("src.inference.predict.read_yaml")
@patch("src.inference.predict.mlflow.sklearn.load_model")
@patch("src.inference.predict.MlflowClient")
@patch("src.inference.predict._load_shap_features")
def test_load_model(
    mock_load_shap, mock_mlflow_client, mock_sklearn_load, mock_read_yaml
):
    mock_read_yaml.return_value = ConfigBox(
        {"model_registry": {"name": "test_model", "stage": "Production"}}
    )

    mock_version_obj = MagicMock()
    mock_version_obj.version = "1"

    mock_client_instance = MagicMock()
    mock_client_instance.search_model_versions.side_effect = lambda query: [
        mock_version_obj
    ]
    mock_mlflow_client.return_value = mock_client_instance

    expected_model = MagicMock()
    mock_sklearn_load.return_value = expected_model

    model = load_model()

    mock_sklearn_load.assert_any_call("models:/test_model/1")
    assert model == expected_model
    mock_load_shap.assert_called()


@patch("src.inference.predict.load_model")
def test_predict(mock_load_model):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_load_model.return_value = mock_model

    input_data = {"feature1": 10, "feature2": "A"}
    prediction, yes_prob, no_prob = predict(input_data)

    assert prediction == 1
    assert yes_prob == 0.8
    assert no_prob == 0.2
    mock_model.predict_proba.assert_called_once()
