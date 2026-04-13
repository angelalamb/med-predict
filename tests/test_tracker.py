"""
Unit tests for tracking/tracker.py.

All tests are offline — no Databricks or MLflow connections are made.
The critical behaviour under test is that the tracker is a safe no-op
when credentials are absent, so it can never break the production API.
"""

from unittest.mock import MagicMock, patch

import pytest

import config
import tracking.tracker as tracker_module
from tracking.tracker import (
    _is_configured,
    log_pipeline_metrics,
    log_query_async,
    pipeline_run,
)


@pytest.fixture(autouse=True)
def reset_configured():
    """Reset the cached _configured flag so each test gets a clean slate."""
    tracker_module._configured = None
    yield
    tracker_module._configured = None


# ---------------------------------------------------------------------------
# _is_configured
# ---------------------------------------------------------------------------


def test_is_configured_result_is_cached(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "https://example.databricks.com")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "dapi123")
    first = _is_configured()
    # Change the underlying value — cached result should not change
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "")
    assert _is_configured() == first


# ---------------------------------------------------------------------------
# pipeline_run
# ---------------------------------------------------------------------------


def test_pipeline_run_yields_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "")
    with pipeline_run("test_run") as run:
        assert run is None


def test_pipeline_run_yields_none_on_mlflow_error(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "https://example.databricks.com")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "dapi123")
    with patch("tracking.tracker._get_mlflow") as mock_get_mlflow:
        mock_get_mlflow.side_effect = ImportError("mlflow not installed")
        with pipeline_run("test_run") as run:
            assert run is None


# ---------------------------------------------------------------------------
# log_pipeline_metrics
# ---------------------------------------------------------------------------


def test_log_pipeline_metrics_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "")
    fake_run = MagicMock()
    with patch("tracking.tracker._get_mlflow") as mock_get_mlflow:
        log_pipeline_metrics(fake_run, {"k_numbers_count": 100.0}, category="ultrasound")
        mock_get_mlflow.assert_not_called()


# ---------------------------------------------------------------------------
# log_query_async
# ---------------------------------------------------------------------------


def test_log_query_async_noop_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "")
    with patch("tracking.tracker.threading.Thread") as mock_thread:
        log_query_async({"cost_usd": 0.01}, None, {"source": "api"})
        mock_thread.assert_not_called()


def test_log_query_async_spawns_daemon_thread_when_configured(monkeypatch):
    monkeypatch.setattr(config, "DATABRICKS_HOST", "https://example.databricks.com")
    monkeypatch.setattr(config, "DATABRICKS_TOKEN", "dapi123")
    with patch("tracking.tracker.threading.Thread") as mock_thread:
        mock_instance = MagicMock()
        mock_thread.return_value = mock_instance
        log_query_async({"cost_usd": 0.01}, None, {"source": "api"})
        mock_thread.assert_called_once()
        # daemon=False so the thread survives process restarts and completes MLflow calls
        _, kwargs = mock_thread.call_args
        assert kwargs.get("daemon") is False
        mock_instance.start.assert_called_once()
