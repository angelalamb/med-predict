"""
Formats the retrieved subgraph into prompt context, calls the LLM API,
and returns a structured analysis response.

The only public function is generate().  Everything else is private
formatting and API machinery.
"""

import time

import anthropic

from config import ANTHROPIC_API_KEY, LLM_MODEL, get_logger
from generation.prompts import get_system_prompt, render_user_prompt

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Anthropic client singleton
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    """
    Return the shared Anthropic client, creating it on first call.

    Returns:
        anthropic.Anthropic client instance.

    Raises:
        RuntimeError: If the API key is not configured.
    """
    global _client

    if _client is not None:
        return _client

    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY is not set in environment")
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not configured. "
            "Set it in your .env file."
        )

    _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    logger.info("Anthropic client initialised (model=%s)", LLM_MODEL)
    return _client


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def _format_single_device(node: dict) -> str:
    """
    Format one device node's properties into a readable text block.

    Args:
        node: Device property dict from the retrieval layer.

    Returns:
        Formatted string block for this device.
    """
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
    ]

    if intended_use:
        lines.append(f"- Intended Use: {intended_use}")
    else:
        lines.append("- Intended Use: Not available")

    return "\n".join(lines)


def _format_edge_summary(edges: list[dict]) -> str:
    """
    Format the predicate edges into a compact summary string.

    Args:
        edges: List of {from_k, to_k} dicts.

    Returns:
        Formatted string listing all predicate relationships.
    """
    if not edges:
        return "No predicate relationships identified within the retrieved subgraph."

    lines = ["**Predicate relationships in retrieved subgraph:**"]
    for edge in edges:
        lines.append(f"- {edge['from_k']} was predicated on {edge['to_k']}")
    return "\n".join(lines)


def _format_device_context(subgraph: dict) -> str:
    """
    Format the full subgraph into a structured context string for the prompt.

    Seeds are listed first, then ancestors, then descendants, each in
    their own section.

    Args:
        subgraph: Dict with 'nodes' and 'edges' keys from the retrieval layer.

    Returns:
        Formatted context string.
    """
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
# Token counting and logging
# ---------------------------------------------------------------------------


def _log_prompt_stats(system: str, user: str) -> None:
    """
    Log approximate token counts for the prompt before sending.

    Uses a rough character-based estimate (4 chars ≈ 1 token) rather
    than a tokeniser, which is sufficient for monitoring purposes.

    Args:
        system: System prompt string.
        user: User prompt string.
    """
    system_tokens = len(system) // 4
    user_tokens = len(user) // 4
    logger.info(
        "Prompt stats — system: ~%d tokens, user: ~%d tokens, total: ~%d tokens",
        system_tokens,
        user_tokens,
        system_tokens + user_tokens,
    )


def _log_response_stats(response: anthropic.types.Message, elapsed: float) -> None:
    """
    Log token usage and timing from the API response.

    Args:
        response: Anthropic Message response object.
        elapsed: Time taken for the API call in seconds.
    """
    usage = response.usage
    logger.info(
        "Generation complete in %.2fs — "
        "input_tokens=%d, output_tokens=%d, stop_reason=%s",
        elapsed,
        usage.input_tokens,
        usage.output_tokens,
        response.stop_reason,
    )


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def _call_llm(system: str, user: str) -> anthropic.types.Message:
    """
    Call the Anthropic messages API with the given prompts.

    Args:
        system: System prompt string.
        user: User prompt string.

    Returns:
        Anthropic Message response object.

    Raises:
        RuntimeError: If the API call fails.
    """
    client = _get_client()

    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response

    except anthropic.AuthenticationError as exc:
        logger.error("Anthropic authentication failed: %s", exc)
        raise RuntimeError("Anthropic API authentication failed") from exc

    except anthropic.RateLimitError as exc:
        logger.error("Anthropic rate limit exceeded: %s", exc)
        raise RuntimeError("Anthropic API rate limit exceeded") from exc

    except anthropic.APIError as exc:
        logger.error("Anthropic API error: %s", exc)
        raise RuntimeError(f"Anthropic API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _extract_text_from_response(response: anthropic.types.Message) -> str:
    """
    Extract the text content from an Anthropic Message response.

    Args:
        response: Anthropic Message object.

    Returns:
        Text content string.

    Raises:
        ValueError: If no text content block is found in the response.
    """
    for block in response.content:
        if block.type == "text":
            return block.text

    logger.error("No text content block found in API response")
    raise ValueError("LLM response contained no text content")


def _build_result(
    query: str,
    analysis_text: str,
    subgraph: dict,
    model: str,
    usage: anthropic.types.Usage,
) -> dict:
    """
    Assemble the final result dict returned to the caller.

    Args:
        query: Original query string.
        analysis_text: Generated analysis text from the LLM.
        subgraph: The retrieved subgraph passed into generation.
        model: Model identifier used for generation.
        usage: Token usage from the API response.

    Returns:
        Structured result dict.
    """
    return {
        "query": query,
        "analysis": analysis_text,
        "subgraph": subgraph,
        "metadata": {
            "model": model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "prompt_version": "v1",
        },
    }


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def generate(query: str, subgraph: dict, prompt_version: str = "v1") -> dict:
    """
    Generate a substantial equivalence analysis for a query device.

    Formats the retrieved subgraph into prompt context, calls the LLM,
    and returns a structured result dict.

    Args:
        query: Natural language description of the device under review.
        subgraph: Retrieved subgraph from retriever.retrieve(), containing
                  'nodes' and 'edges' keys.
        prompt_version: Prompt template version to use (default 'v1').

    Returns:
        Dict with keys:
          'query'    — original query string
          'analysis' — full LLM-generated analysis text (markdown)
          'subgraph' — the subgraph passed in (passed through for the UI)
          'metadata' — dict with model, token counts, prompt_version

    Raises:
        RuntimeError: If the LLM API call fails.
        ValueError: If the subgraph contains no usable nodes.
    """
    nodes = subgraph.get("nodes", [])

    if not nodes:
        logger.warning(
            "generate() called with empty subgraph for query: %r", query[:80]
        )
        raise ValueError(
            "Cannot generate analysis: subgraph contains no device nodes. "
            "Check that the retrieval step returned results."
        )

    logger.info(
        "Generating analysis | nodes=%d | edges=%d | model=%s | query=%r",
        len(nodes),
        len(subgraph.get("edges", [])),
        LLM_MODEL,
        query[:80],
    )

    system = get_system_prompt(version=prompt_version)
    device_context = _format_device_context(subgraph)
    user = render_user_prompt(
        query=query,
        device_context=device_context,
        version=prompt_version,
    )

    _log_prompt_stats(system, user)

    start = time.time()
    response = _call_llm(system, user)
    elapsed = time.time() - start

    _log_response_stats(response, elapsed)

    analysis_text = _extract_text_from_response(response)

    result = _build_result(
        query=query,
        analysis_text=analysis_text,
        subgraph=subgraph,
        model=LLM_MODEL,
        usage=response.usage,
    )

    logger.info("Analysis generation complete for query: %r", query[:80])

    return result
