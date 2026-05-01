"""
Calls the Claude API and returns a generated answer with token usage.

Two public interfaces:
  - Generator class   — used by the API (routes.py)
  - generate()        — used by the Streamlit app (streamlit_app.py)
"""

import os
import time
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, get_logger
from generation.prompts import IRRELEVANT_QUERY_SENTINEL, get_system_prompt, render_user_prompt

logger = get_logger(__name__)


class IrrelevantQueryError(Exception):
    """Raised when the LLM determines the query is not about a medical device."""


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_single_device(node: dict) -> str:
    k_number = node.get("k_number", "Unknown")
    device_name = node.get("device_name", "Unknown")
    applicant = node.get("applicant", "Unknown")
    product_code = node.get("product_code", "Unknown")
    decision_date = node.get("decision_date", "Unknown")
    intended_use = node.get("intended_use", "").strip()
    direction = node.get("direction", "unknown")
    is_seed = node.get("is_seed", False)

    tag = "SEED (direct semantic match)" if is_seed else direction.upper()

    lines = [
        f"### {device_name} ({k_number}) [{tag}]",
        f"- Applicant: {applicant}",
        f"- Product Code: {product_code}",
        f"- Decision Date: {decision_date}",
        f"- Intended Use: {intended_use or 'Not available'}",
    ]
    return "\n".join(lines)


def _format_edge_summary(edges: list[dict]) -> str:
    if not edges:
        return "No predicate relationships identified within the retrieved subgraph."
    lines = ["**Predicate relationships in retrieved subgraph:**"]
    for edge in edges:
        lines.append(f"- {edge['from_k']} was predicated on {edge['to_k']}")
    return "\n".join(lines)


def _format_device_context(subgraph: dict) -> str:
    """Format a subgraph dict into a structured context string for the prompt."""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    seeds = [n for n in nodes if n.get("is_seed")]
    ancestors = [n for n in nodes if not n.get("is_seed") and n.get("direction") == "ancestor"]
    descendants = [n for n in nodes if not n.get("is_seed") and n.get("direction") == "descendant"]

    sections = []
    if seeds:
        sections.append("## Seed Devices (Semantic Matches)\n")
        sections.extend(_format_single_device(n) for n in seeds)
    if ancestors:
        sections.append("\n## Ancestor Devices (Upstream Predicates)\n")
        sections.extend(_format_single_device(n) for n in ancestors)
    if descendants:
        sections.append("\n## Descendant Devices (Downstream Citations)\n")
        sections.extend(_format_single_device(n) for n in descendants)

    sections.append(f"\n## Predicate Network\n\n{_format_edge_summary(edges)}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


def _get_client() -> anthropic.Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. Set it in your .env file."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Generator class — used by the API
# ---------------------------------------------------------------------------


class Generator:
    """Generate answers using the Claude API with token usage tracking."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = LLM_MODEL

    def generate(self, query: str, context: list) -> str:
        """Generate an answer, discarding token usage."""
        answer, _ = self.generate_with_usage(query, context)
        return answer

    def generate_with_usage(self, query: str, context: list) -> tuple[str, dict[str, int]]:
        """
        Generate an answer and return token usage for cost tracking.

        Args:
            query: Natural language device description.
            context: List of device node dicts from the retrieval layer.

        Returns:
            Tuple of (answer: str, tokens: dict) where tokens has
            keys 'input' and 'output'.
        """
        subgraph = {"nodes": context, "edges": []}
        device_context = _format_device_context(subgraph)
        system = get_system_prompt()
        user = render_user_prompt(query=query, device_context=device_context)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            raise

        answer = response.content[0].text.strip()
        if answer == IRRELEVANT_QUERY_SENTINEL:
            raise IrrelevantQueryError()
        tokens = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
        }
        return answer, tokens


# ---------------------------------------------------------------------------
# Module-level generate() — used by the Streamlit app
# ---------------------------------------------------------------------------


def generate(query: str, subgraph: dict, prompt_version: str = "v1") -> dict:
    """
    Generate a substantial equivalence analysis for a query device.

    Args:
        query: Natural language description of the device under review.
        subgraph: Retrieved subgraph from retriever.retrieve(), containing
                  'nodes' and 'edges' keys.
        prompt_version: Prompt template version to use (default 'v1').

    Returns:
        Dict with keys:
          'query'    — original query string
          'analysis' — LLM-generated analysis text (markdown)
          'subgraph' — the subgraph passed in (passed through for the UI)
          'metadata' — dict with model, token counts, prompt_version

    Raises:
        RuntimeError: If the LLM API call fails.
        ValueError: If the subgraph contains no usable nodes.
    """
    nodes = subgraph.get("nodes", [])

    if not nodes:
        logger.warning("generate() called with empty subgraph")
        raise ValueError(
            "Cannot generate analysis: subgraph contains no device nodes. "
            "Check that the retrieval step returned results."
        )

    logger.info(
        "Generating analysis | nodes=%d | edges=%d | model=%s",
        len(nodes),
        len(subgraph.get("edges", [])),
        LLM_MODEL,
    )

    client = _get_client()
    system = get_system_prompt(version=prompt_version)
    device_context = _format_device_context(subgraph)
    user = render_user_prompt(query=query, device_context=device_context, version=prompt_version)

    try:
        start = time.time()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        elapsed = time.time() - start
    except anthropic.AuthenticationError as exc:
        logger.error("Anthropic authentication failed: %s", exc)
        raise RuntimeError("Anthropic API authentication failed") from exc
    except anthropic.RateLimitError as exc:
        logger.error("Anthropic rate limit exceeded: %s", exc)
        raise RuntimeError("Anthropic API rate limit exceeded") from exc
    except anthropic.APIError as exc:
        logger.error("Anthropic API error: %s", exc)
        raise RuntimeError(f"Anthropic API error: {exc}") from exc

    logger.info(
        "Generation complete in %.2fs — input_tokens=%d, output_tokens=%d",
        elapsed,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    analysis_text = next(
        (block.text for block in response.content if block.type == "text"), None
    )
    if analysis_text is None:
        raise ValueError("LLM response contained no text content")

    return {
        "query": query,
        "analysis": analysis_text,
        "subgraph": subgraph,
        "metadata": {
            "model": LLM_MODEL,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "prompt_version": prompt_version,
        },
    }
