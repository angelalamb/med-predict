"""
MLflow tracking wrapper for MedPredict.

This is the only module in the project that imports mlflow. All other modules
call this wrapper. If Databricks credentials are absent, every method is a
silent no-op — tracking must never crash or slow down the application.
"""

import threading
import time
from contextlib import contextmanager
from typing import Optional

import config
from config import get_logger

logger = get_logger(__name__)

# Cached once on first call to _is_configured()
_configured: Optional[bool] = None


def _is_configured() -> bool:
    global _configured
    if _configured is None:
        _configured = bool(config.DATABRICKS_HOST and config.DATABRICKS_TOKEN)
        logger.debug("MLflow configured: %s", _configured)
        if not _configured:
            logger.debug(
                "MLflow tracking disabled: DATABRICKS_HOST or DATABRICKS_TOKEN not set"
            )
    return _configured


def _get_mlflow():
    """Lazy import so mlflow is never loaded when tracking is disabled."""
    import mlflow  # noqa: PLC0415
    return mlflow


def _setup_tracking(mlflow) -> None:
    """Configure the Databricks tracking URI. Called before every MLflow operation."""
    mlflow.set_tracking_uri("databricks")


def _get_or_create_experiment(mlflow, name: str) -> str:
    """Return the experiment_id for *name*, creating it if it does not exist."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        return mlflow.create_experiment(name)
    return experiment.experiment_id


# ---------------------------------------------------------------------------
# Pipeline tracking
# ---------------------------------------------------------------------------


@contextmanager
def pipeline_run(run_name: str):
    """
    Context manager for pipeline tracking.

    Yields the active MLflow run so the caller can pass it to
    log_pipeline_metrics(). Yields None if tracking is not configured.

    Usage::

        with tracker.pipeline_run("pipeline_20240101") as run:
            # ... pipeline steps ...
            tracker.log_pipeline_metrics(run, {...})
    """
    if not _is_configured():
        yield None
        return

    try:
        mlflow = _get_mlflow()
        _setup_tracking(mlflow)
        exp_id = _get_or_create_experiment(mlflow, config.MLFLOW_EXPERIMENT_PIPELINE)
        with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
            mlflow.set_tags(
                {
                    "embedding_model": config.EMBEDDING_MODEL_NAME,
                    "llm_model": config.LLM_MODEL,
                    "prompt_version": config.PROMPT_VERSION,
                }
            )
            yield run
    except Exception:
        logger.debug("MLflow pipeline_run context failed", exc_info=True)
        yield None


def log_pipeline_metrics(run, metrics: dict) -> None:
    """
    Log pipeline metrics to an active run. No-op if *run* is None.

    Args:
        run:     The MLflow run object yielded by pipeline_run().
        metrics: Numeric metrics dict, e.g. {"k_numbers_count": 312.0, ...}.
    """
    if run is None or not _is_configured():
        return
    try:
        mlflow = _get_mlflow()
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "product_codes": ",".join(config.ULTRASOUND_PRODUCT_CODES),
                "min_year": str(config.MIN_SUBMISSION_YEAR),
                "prompt_version": config.PROMPT_VERSION,
            }
        )
    except Exception:
        logger.debug("MLflow log_pipeline_metrics failed", exc_info=True)


# ---------------------------------------------------------------------------
# Query tracking (async, sampled)
# ---------------------------------------------------------------------------


def log_query_async(
    metrics: dict,
    judge_input: Optional[dict],
    tags: dict,
) -> None:
    """
    Log query metrics to MLflow in a background daemon thread.

    Never blocks the API response. Errors inside the thread are swallowed
    at DEBUG level.

    Args:
        metrics:     Numeric metrics (latency, tokens, cost, retrieval counts).
        judge_input: If provided, runs LLM-as-judge and logs the 5 scores into
                     the same MLflow run. Must contain keys:
                       - "query"             (str)
                       - "retrieved_devices" (list[dict])
                       - "analysis"          (str)
                     Pass None to skip judge evaluation for this sample.
        tags:        String metadata (model version, prompt version, k, source).
    """
    if not _is_configured():
        return

    thread = threading.Thread(
        target=_log_query,
        args=(metrics, judge_input, tags),
        daemon=True,
    )
    thread.start()


def _log_query(metrics: dict, judge_input: Optional[dict], tags: dict) -> None:
    """Internal: write one query sample to MLFLOW_EXPERIMENT_QUERY."""
    try:
        mlflow = _get_mlflow()
        _setup_tracking(mlflow)
        exp_id = _get_or_create_experiment(mlflow, config.MLFLOW_EXPERIMENT_QUERY)

        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)

            if judge_input is not None:
                _run_and_log_judge(mlflow, judge_input)

    except Exception:
        logger.debug("MLflow _log_query failed", exc_info=True)


def _run_and_log_judge(mlflow, judge_input: dict) -> None:
    """Run LLM-as-judge and log the 5 scores into the current active run."""
    try:
        from eval.judge import LLMJudge  # noqa: PLC0415 — intentional lazy import

        judge = LLMJudge()
        scores = judge.evaluate(
            query=judge_input["query"],
            retrieved_devices=judge_input["retrieved_devices"],
            analysis=judge_input["analysis"],
        )
        if scores:
            mlflow.log_metrics({f"judge_{k}": v for k, v in scores.items()})
    except Exception:
        logger.debug("MLflow judge evaluation failed", exc_info=True)
