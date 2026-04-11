"""
Unit tests for eval/judge.py.

All tests are offline — no Anthropic API calls are made.
Focus areas:
  - _extract_scores: the JSON parsing logic that is most likely to fail in prod
  - _build_judge_prompt: truncation of devices and analysis
  - LLMJudge.evaluate: graceful handling of API and parse failures
"""

from unittest.mock import MagicMock

import pytest

import config
from eval.judge import LLMJudge, _build_judge_prompt, _extract_scores


# ---------------------------------------------------------------------------
# _extract_scores
# ---------------------------------------------------------------------------


def test_extract_scores_parses_clean_json():
    text = (
        '{"retrieval_relevance": 4, "analysis_completeness": 5, '
        '"factual_grounding": 3, "regulatory_reasoning": 4, "actionability": 4}'
    )
    scores = _extract_scores(text)
    assert scores["retrieval_relevance"] == 4.0
    assert scores["analysis_completeness"] == 5.0
    assert scores["factual_grounding"] == 3.0
    assert scores["regulatory_reasoning"] == 4.0
    assert scores["actionability"] == 4.0


def test_extract_scores_parses_json_after_reasoning_text():
    text = (
        "The retrieved devices are mostly relevant but regulatory reasoning "
        "has a minor gap in predicate direction.\n"
        '{"retrieval_relevance": 4, "analysis_completeness": 3, '
        '"factual_grounding": 4, "regulatory_reasoning": 3, "actionability": 3}'
    )
    scores = _extract_scores(text)
    assert len(scores) == 5
    assert scores["regulatory_reasoning"] == 3.0


def test_extract_scores_raises_when_no_json_present():
    with pytest.raises(ValueError, match="No JSON object found"):
        _extract_scores("The analysis looks reasonable overall.")


def test_extract_scores_ignores_unknown_criteria():
    # Extra keys the judge invents should not appear in output
    text = (
        '{"retrieval_relevance": 4, "analysis_completeness": 5, '
        '"factual_grounding": 3, "regulatory_reasoning": 4, '
        '"actionability": 4, "unknown_criterion": 2}'
    )
    scores = _extract_scores(text)
    assert "unknown_criterion" not in scores


def test_extract_scores_handles_partial_criteria():
    # If the judge only returns some keys, we get what we can
    text = '{"retrieval_relevance": 4, "actionability": 3}'
    scores = _extract_scores(text)
    assert scores["retrieval_relevance"] == 4.0
    assert scores["actionability"] == 3.0
    assert "analysis_completeness" not in scores


def test_extract_scores_parses_json_with_internal_newlines():
    # The model may format the JSON across multiple lines
    text = (
        "Brief reasoning here.\n"
        "{\n"
        '  "retrieval_relevance": 4,\n'
        '  "analysis_completeness": 5,\n'
        '  "factual_grounding": 3,\n'
        '  "regulatory_reasoning": 4,\n'
        '  "actionability": 4\n'
        "}"
    )
    scores = _extract_scores(text)
    assert len(scores) == 5
    assert scores["analysis_completeness"] == 5.0


def test_extract_scores_accepts_float_values_from_model():
    # Models sometimes emit "4.0" instead of 4 — clamping must still work
    text = (
        '{"retrieval_relevance": 4.0, "analysis_completeness": 5.0, '
        '"factual_grounding": 3.0, "regulatory_reasoning": 4.0, "actionability": 4.0}'
    )
    scores = _extract_scores(text)
    assert scores["retrieval_relevance"] == 4.0
    assert scores["factual_grounding"] == 3.0


# ---------------------------------------------------------------------------
# _build_judge_prompt
# ---------------------------------------------------------------------------


def _make_devices(n: int) -> list[dict]:
    return [
        {
            "k_number": f"K{i:06d}",
            "device_name": "Test Device",
            "direction": "seed",
            "similarity_score": 0.9,
        }
        for i in range(n)
    ]


def test_build_judge_prompt_truncates_devices_to_max():
    devices = _make_devices(config.JUDGE_MAX_DEVICES + 5)
    prompt = _build_judge_prompt("query", devices, "analysis")
    # Only JUDGE_MAX_DEVICES K-numbers should appear
    k_count = sum(1 for d in devices if d["k_number"] in prompt)
    assert k_count <= config.JUDGE_MAX_DEVICES


def test_build_judge_prompt_truncates_long_analysis():
    long_analysis = "X" * (config.JUDGE_MAX_ANALYSIS_CHARS + 200)
    prompt = _build_judge_prompt("query", [], long_analysis)
    assert "[analysis truncated]" in prompt


def test_build_judge_prompt_does_not_truncate_short_analysis():
    short_analysis = "Short analysis."
    prompt = _build_judge_prompt("query", [], short_analysis)
    assert "[analysis truncated]" not in prompt
    assert short_analysis in prompt


# ---------------------------------------------------------------------------
# LLMJudge.evaluate
# ---------------------------------------------------------------------------


def _make_judge(mock_response_text: str) -> LLMJudge:
    """Build a LLMJudge with a mocked Anthropic client."""
    judge = LLMJudge.__new__(LLMJudge)
    mock_content = MagicMock()
    mock_content.text = mock_response_text
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    judge.client = mock_client
    return judge


def test_evaluate_returns_five_scores_on_success():
    valid_json = (
        '{"retrieval_relevance": 4, "analysis_completeness": 5, '
        '"factual_grounding": 3, "regulatory_reasoning": 4, "actionability": 4}'
    )
    judge = _make_judge(valid_json)
    result = judge.evaluate("query", _make_devices(3), "analysis text")
    assert len(result) == 5
    assert all(1.0 <= v <= 5.0 for v in result.values())


def test_evaluate_returns_empty_dict_on_api_error(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    judge = LLMJudge.__new__(LLMJudge)
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("connection error")
    judge.client = mock_client
    result = judge.evaluate("query", [], "analysis")
    assert result == {}


def test_evaluate_returns_empty_dict_on_malformed_json(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key")
    judge = _make_judge("I cannot score this analysis.")
    result = judge.evaluate("query", [], "analysis")
    assert result == {}


def test_evaluate_uses_configured_judge_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MODEL", "claude-test-model")
    valid_json = (
        '{"retrieval_relevance": 3, "analysis_completeness": 3, '
        '"factual_grounding": 3, "regulatory_reasoning": 3, "actionability": 3}'
    )
    judge = _make_judge(valid_json)
    judge.evaluate("query", [], "analysis")
    call_kwargs = judge.client.messages.create.call_args[1]
    assert call_kwargs["model"] == "claude-test-model"


def test_evaluate_respects_judge_max_tokens(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_MAX_TOKENS", 256)
    valid_json = (
        '{"retrieval_relevance": 3, "analysis_completeness": 3, '
        '"factual_grounding": 3, "regulatory_reasoning": 3, "actionability": 3}'
    )
    judge = _make_judge(valid_json)
    judge.evaluate("query", [], "analysis")
    call_kwargs = judge.client.messages.create.call_args[1]
    assert call_kwargs["max_tokens"] == 256
