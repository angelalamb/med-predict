"""
LLM-as-judge evaluation for MedPredict query responses.

Evaluates generated 510(k) substantial equivalence analyses on five criteria
using Claude. Scores are integers 1–5 per criterion. Query content and
analysis text are never persisted — evaluation is entirely in-memory.
"""

import json
import re

import anthropic

import config
from config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Criteria registry
# ---------------------------------------------------------------------------

CRITERIA = {
    "retrieval_relevance": (
        "Are the retrieved devices topically appropriate for the query's "
        "indication and device category?"
    ),
    "analysis_completeness": (
        "Does the response explicitly address both (1) intended use comparison "
        "AND (2) technological characteristics for the top predicate candidates?"
    ),
    "factual_grounding": (
        "Are all substantive claims tied to specific K-numbers present in the "
        "retrieved device set? No invented or uncited claims?"
    ),
    "regulatory_reasoning": (
        "Is the substantial equivalence pathway logic sound — correct predicate "
        "direction, plausible clearance pathway, appropriate use of "
        "PREDICATED_ON relationships?"
    ),
    "actionability": (
        "Could a regulatory affairs professional use this output to draft a "
        "510(k) SE argument without major revision?"
    ),
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = """You are a FDA regulatory affairs reviewer evaluating \
an AI assistant's 510(k) predicate device analysis.

Score each criterion from 1 to 5 using the rubrics below.

retrieval_relevance
  1 = Most retrieved devices are from unrelated indications or product categories
  5 = All retrieved devices are from the same or directly related indication \
and product category

analysis_completeness
  1 = Missing both SE components, or addresses only one superficially
  5 = Intended use match AND technological characteristics explicitly addressed \
for each ranked candidate

factual_grounding
  1 = Claims are unsupported, vague, or cite devices not in the retrieved set
  5 = Every substantive claim cites a specific K-number from the retrieved \
device list

regulatory_reasoning
  1 = SE direction is inverted, predicates post-date the subject, or pathway \
is implausible
  5 = Correct predicate direction, timeline is sound, pathway is clearly \
articulable from the data

actionability
  1 = Requires complete rewrite; not useful in current form
  5 = Could be used directly to draft an SE argument with only minor edits

First reason briefly (2–3 sentences total across all criteria). Then end your \
response with a JSON object on a single line:
{"retrieval_relevance": <int>, "analysis_completeness": <int>, \
"factual_grounding": <int>, "regulatory_reasoning": <int>, "actionability": <int>}
"""


def _build_judge_prompt(
    query: str,
    retrieved_devices: list[dict],
    analysis: str,
) -> str:
    device_lines = "\n".join(
        "- {k} | {name} | {direction} | similarity={sim}".format(
            k=d.get("k_number", "?"),
            name=d.get("device_name", "?"),
            direction=d.get("direction", "?"),
            sim=d.get("similarity_score", "N/A"),
        )
        for d in retrieved_devices[: config.JUDGE_MAX_DEVICES]
    )

    truncated_analysis = analysis[: config.JUDGE_MAX_ANALYSIS_CHARS]
    if len(analysis) > config.JUDGE_MAX_ANALYSIS_CHARS:
        truncated_analysis += "\n[analysis truncated]"

    criteria_lines = "\n".join(
        f"{i + 1}. {name}: {desc}"
        for i, (name, desc) in enumerate(CRITERIA.items())
    )

    return (
        f"## Query\n{query}\n\n"
        f"## Retrieved Devices ({len(retrieved_devices)} total)\n{device_lines}\n\n"
        f"## Generated Analysis\n{truncated_analysis}\n\n"
        f"## Evaluation Criteria\n{criteria_lines}"
    )


def _extract_scores(text: str) -> dict[str, float]:
    """
    Extract the JSON scores object from the judge response text.

    The model may prepend reasoning before the JSON, so we search for the
    last JSON object in the response rather than parsing the full text.

    Raises:
        ValueError: If no valid JSON object is found.
    """
    matches = re.findall(r"\{[^{}]+\}", text, re.DOTALL)
    if not matches:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    scores = json.loads(matches[-1])
    return {
        k: float(min(5, max(1, scores[k])))
        for k in CRITERIA
        if k in scores
    }


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluates a query/response pair using Claude as the judge."""

    def __init__(self) -> None:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def evaluate(
        self,
        query: str,
        retrieved_devices: list[dict],
        analysis: str,
    ) -> dict[str, float]:
        """
        Score a query/response pair on the five evaluation criteria.

        Args:
            query:             Original natural language query.
            retrieved_devices: List of device node dicts from the retriever.
            analysis:          LLM-generated substantial equivalence analysis.

        Returns:
            Dict mapping criterion name → score (float 1–5). Returns an empty
            dict if the evaluation fails so callers can handle gracefully.
        """
        prompt = _build_judge_prompt(query, retrieved_devices, analysis)

        try:
            response = self.client.messages.create(
                model=config.JUDGE_MODEL,
                max_tokens=config.JUDGE_MAX_TOKENS,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            return _extract_scores(text)

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Judge response could not be parsed: %s", exc)
            return {}
        except Exception as exc:
            logger.warning("Judge evaluation failed: %s", exc)
            return {}
