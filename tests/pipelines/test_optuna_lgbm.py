from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.pipelines.train_optuna_lgbm import objective


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "tenure": np.random.randint(0, 72, n),
            "MonthlyCharges": np.random.uniform(18, 120, n),
            "TotalCharges": np.random.uniform(0, 8000, n),
            "gender": np.random.choice(["Male", "Female"], n),
            "Partner": np.random.choice(["Yes", "No"], n),
            "Dependents": np.random.choice(["Yes", "No"], n),
            "PhoneService": np.random.choice(["Yes", "No"], n),
            "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": np.random.choice(
                ["Yes", "No", "No internet service"], n
            ),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
            ),
            "SeniorCitizen": np.random.choice([0, 1], n),
            "Churn": np.random.choice([0, 1], n, p=[0.73, 0.27]),
        }
    )
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return df, X, y


@patch("src.pipelines.train_optuna_lgbm.mlflow")
def test_lgbm_objective_returns_f1(mock_mlflow, sample_data):
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    df, X, y = sample_data
    split = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    trial = MagicMock()
    trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
    trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

    result = objective(trial, X_train, y_train, X_val, y_val, df, "Churn")

    assert isinstance(result, float), "Objective must return a float F1 score"
    assert 0.0 <= result <= 1.0, f"F1 score out of range: {result}"


@patch("src.pipelines.train_optuna_lgbm.mlflow")
def test_lgbm_objective_handles_imbalance(mock_mlflow, sample_data):
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    df, X, y = sample_data
    split = int(0.8 * len(X))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    captured_params = {}

    trial = MagicMock()
    trial.suggest_int.side_effect = lambda name, low, high: (low + high) // 2
    trial.suggest_float.side_effect = lambda name, low, high, **kw: (low + high) / 2
    trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

    objective(trial, X_train, y_train, X_val, y_val, df, "Churn")

    call_args = mock_mlflow.start_run.return_value.__enter__.return_value
