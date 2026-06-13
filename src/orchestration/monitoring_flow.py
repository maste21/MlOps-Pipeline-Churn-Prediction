import os

import psycopg
from prefect import flow, task
from prefect.utilities.annotations import allow_failure

from src.monitoring.evidently_drift import run_evidently_monitoring
from src.utils.logger import logger

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))


def _get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "db"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "example"),
        autocommit=True,
    )


def _read_latest_drift_score() -> float | None:
    sql = """
        SELECT report_json -> 'metrics' AS metrics
        FROM   evidently_reports
        ORDER  BY timestamp DESC
        LIMIT  1;
    """
    try:
        with _get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                if row is None:
                    logger.warning("No evidently reports found in DB.")
                    return None

                metrics = row[0]
                for metric in metrics:
                    if metric.get("metric") == "DatasetDriftMetric":
                        share = metric["result"].get("share_of_drifted_columns")
                        if share is not None:
                            return float(share)

                logger.warning("DatasetDriftMetric not found in latest report.")
                return None

    except Exception as exc:
        logger.error("Failed to read drift score from DB: %s", exc)
        return None


@task(name="run-monitoring")
def run_monitoring_task():
    run_evidently_monitoring()


@task(name="check-drift-threshold")
def check_drift_threshold() -> bool:
    score = _read_latest_drift_score()
    if score is None:
        logger.warning("Could not determine drift score — skipping retraining check.")
        return False

    logger.info(
        "Latest drift score: %.4f  (threshold: %.4f)",
        score,
        DRIFT_THRESHOLD,
    )
    if score >= DRIFT_THRESHOLD:
        logger.warning(
            "⚠️  Drift score %.4f >= threshold %.4f — retraining will be triggered.",
            score,
            DRIFT_THRESHOLD,
        )
        return True

    logger.info("✅ Drift score below threshold — no retraining needed.")
    return False


@task(name="trigger-retraining")
def trigger_retraining(should_retrain: bool):
    if not should_retrain:
        logger.info("Retraining skipped — drift within acceptable range.")
        return

    logger.info("🚀 Drift threshold exceeded — launching training pipeline …")
    from src.orchestration.train_flow import training_pipeline

    training_pipeline()
    logger.info("✅ Drift-triggered retraining completed.")


@flow(name="monitoring-pipeline")
def monitoring_pipeline():
    run_monitoring_task()
    should_retrain = check_drift_threshold()
    trigger_retraining(should_retrain)


if __name__ == "__main__":
    monitoring_pipeline()
