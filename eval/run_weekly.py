"""
Weekly evaluation summary script.

Reads all query_metrics runs logged in the past JUDGE_LOOKBACK_DAYS days,
computes aggregate statistics, and writes a summary run to the
llm_judge_eval experiment.

Logs a WARNING for any judge criterion whose weekly average falls below
JUDGE_SCORE_PASS_THRESHOLD.

Run from the project root::

    python -m eval.run_weekly
"""

import sys
from datetime import datetime, timedelta, timezone

import config
from config import get_logger

logger = get_logger(__name__)

_OPERATIONAL_METRICS = [
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "generation_latency_ms",
    "retrieval_latency_ms",
]

_JUDGE_METRIC_KEYS = [f"judge_{k}" for k in (
    "retrieval_relevance",
    "analysis_completeness",
    "factual_grounding",
    "regulatory_reasoning",
    "actionability",
)]


def run_weekly_eval() -> None:
    if not (config.DATABRICKS_HOST and config.DATABRICKS_TOKEN):
        logger.error(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set to run the weekly eval"
        )
        sys.exit(1)

    import mlflow

    mlflow.set_tracking_uri("databricks")

    cutoff = datetime.now(timezone.utc) - timedelta(days=config.JUDGE_LOOKBACK_DAYS)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    query_experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_QUERY)
    if query_experiment is None:
        logger.error(
            "Experiment %s not found — has the API received any sampled queries yet?",
            config.MLFLOW_EXPERIMENT_QUERY,
        )
        sys.exit(1)

    runs = mlflow.search_runs(
        experiment_ids=[query_experiment.experiment_id],
        filter_string=f"attributes.start_time >= {cutoff_ms}",
        output_format="pandas",
    )

    if runs.empty:
        logger.info(
            "No query runs found in the past %d days — nothing to summarise",
            config.JUDGE_LOOKBACK_DAYS,
        )
        return

    total_runs = len(runs)

    # Subset that includes at least one judge metric
    first_judge_col = f"metrics.{_JUDGE_METRIC_KEYS[0]}"
    judge_runs = (
        runs.dropna(subset=[first_judge_col])
        if first_judge_col in runs.columns
        else runs.iloc[0:0]  # empty frame with same columns
    )

    summary: dict[str, float] = {
        "total_query_runs": float(total_runs),
        "judge_evaluated_runs": float(len(judge_runs)),
    }

    # Judge score aggregates
    for key in _JUDGE_METRIC_KEYS:
        col = f"metrics.{key}"
        if col in judge_runs.columns and not judge_runs[col].dropna().empty:
            summary[f"avg_{key}"] = float(judge_runs[col].mean())
            summary[f"min_{key}"] = float(judge_runs[col].min())

    # Operational metric aggregates
    for metric in _OPERATIONAL_METRICS:
        col = f"metrics.{metric}"
        if col in runs.columns and not runs[col].dropna().empty:
            summary[f"avg_{metric}"] = float(runs[col].mean())

    # Cost and token totals are more useful as sums
    for metric in ("input_tokens", "output_tokens", "cost_usd"):
        col = f"metrics.{metric}"
        if col in runs.columns and not runs[col].dropna().empty:
            summary[f"total_{metric}"] = float(runs[col].sum())

    # Write summary run
    eval_experiment = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_EVAL)
    eval_exp_id = (
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT_EVAL)
        if eval_experiment is None
        else eval_experiment.experiment_id
    )

    week_label = cutoff.strftime("%Y-W%W")
    with mlflow.start_run(experiment_id=eval_exp_id, run_name=f"weekly_{week_label}"):
        mlflow.log_metrics(summary)
        mlflow.set_tags(
            {
                "period_start": cutoff.isoformat(),
                "period_end": datetime.now(timezone.utc).isoformat(),
                "judge_model": config.JUDGE_MODEL,
                "lookback_days": str(config.JUDGE_LOOKBACK_DAYS),
            }
        )

    logger.info("Weekly eval summary logged for %s: %s", week_label, summary)

    # Quality gate
    for key in _JUDGE_METRIC_KEYS:
        avg_key = f"avg_{key}"
        if avg_key in summary and summary[avg_key] < config.JUDGE_SCORE_PASS_THRESHOLD:
            logger.warning(
                "Quality threshold breach: %s avg=%.2f (threshold=%.1f)",
                key,
                summary[avg_key],
                config.JUDGE_SCORE_PASS_THRESHOLD,
            )


if __name__ == "__main__":
    run_weekly_eval()
