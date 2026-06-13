.PHONY: all ingest split train_logreg train_rf train_xgb train_lgbm train_mlp \
        train_all train_ensemble evaluate test quality_checks build \
        integration_test publish setup precommit_check prefect_deploy orchestration

# --- ML PIPELINE ---

# Run the full pipeline: ingest -> split -> all models -> ensemble -> evaluate
all: ingest split train_all train_ensemble evaluate

# Download or ingest raw data
ingest:
	PYTHONPATH=. python src/pipelines/data_ingestion.py

# Split data into train/validation/test sets
split:
	PYTHONPATH=. python src/pipelines/data_split.py

# Train Logistic Regression with Optuna and track with MLflow
train_logreg:
	PYTHONPATH=. python src/pipelines/train_optuna_lr.py

# Train Random Forest with Optuna and track with MLflow
train_rf:
	PYTHONPATH=. python src/pipelines/train_optuna_rf.py

# Train XGBoost with Optuna and track with MLflow
train_xgb:
	PYTHONPATH=. python src/pipelines/train_optuna_xgb.py

# Train LightGBM with Optuna and track with MLflow
train_lgbm:
	PYTHONPATH=. python src/pipelines/train_optuna_lgbm.py

# Train MLP Neural Network with Optuna and track with MLflow
train_mlp:
	PYTHONPATH=. python src/pipelines/train_optuna_mlp.py

# Run all base model training scripts sequentially
train_all: train_logreg train_rf train_xgb train_lgbm train_mlp

train_ensemble:
	PYTHONPATH=. python src/pipelines/train_ensemble.py

# Evaluate best model (ensemble-voting-model) on the held-out test set
evaluate:
	PYTHONPATH=. python src/pipelines/evaluate_model.py

orchestration:
	PYTHONPATH=. python src/orchestration/create_deployment.py

# --- DEV UTILS ---

# Run tests (unit + integration)
test:
	pytest tests/

# Format and lint
quality_checks:
	black --check src/ tests/
	isort --check-only src/ tests/
	pylint --recursive=y src/ tests/

# Run what happens in pre-commit
precommit_check: quality_checks test

# Prefect deployment
prefect_deploy:
	PYTHONPATH=. python src/orchestration/create_deployment.py

# Initial setup
setup:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install