"""
generation/prompts.py

All prompt templates used by the generation layer.

Templates are versioned string constants.  The generator imports from
here rather than constructing prompts inline, keeping prompt logic
separate from API call logic and making iteration straightforward.
"""

# ---------------------------------------------------------------------------
# V1 Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_V1 = """You are a regulatory affairs assistant specialising in \
FDA 510(k) premarket notifications for medical devices.

Your role is to analyse candidate predicate devices and produce a structured \
substantial equivalence assessment grounded exclusively in the device data \
provided to you.

Rules you must follow:
- Ground every claim in the provided device data. Do not use general \
knowledge about device types unless it is confirmed by the data.
- Cite the K-number when referencing any specific device.
- Structure your analysis around the FDA's two-part substantial equivalence \
test: (1) same intended use, and (2) same or equivalent technological \
characteristics.
- If the data is insufficient to assess a criterion, say so explicitly rather \
than speculating.
- Rank candidates from strongest to weakest predicate fit based on the \
available evidence.
- Use precise, professional regulatory language appropriate for a 510(k) \
submission context.
"""

USER_PROMPT_TEMPLATE_V1 = """## Query Device Description

{query}

---

## Retrieved Device Data

The following devices were retrieved from the FDA 510(k) database as \
potentially relevant. Seed devices were identified by semantic similarity \
to the query. Ancestor and descendant devices were identified by traversing \
the predicate network.

{device_context}

---

## Task

Produce a substantial equivalence analysis with the following structure:

### Ranked Predicate Candidates

For each candidate (ranked strongest to weakest):

**[Rank]. [Device Name] ([K-number])**
- Applicant: [applicant name]
- Cleared: [decision date]
- Intended Use Match: [assessment — identical / substantially similar / \
partially similar / not similar]
- Technological Characteristics: [assessment based on available data]
- Equivalence Argument: [2–4 sentences grounding the equivalence claim \
in the retrieved data]
- Limitations: [any gaps in the available data that limit this assessment]

---

### Predicate Network Notes

Briefly describe any notable patterns in the predicate network — for example, \
whether candidates share a common ancestor, whether there is a clear lineage \
chain, or whether any candidates appear isolated.

---

### Recommended Next Steps

List 2–3 concrete actions a regulatory affairs team should take to validate \
the top candidate(s) before submitting.
"""


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def render_user_prompt(query: str, device_context: str, version: str = "v1") -> str:
    """
    Render the user prompt template with query and device context.

    Args:
        query: Natural language device description from the user.
        device_context: Formatted string of retrieved device data.
        version: Prompt version to use. Currently only 'v1' is supported.

    Returns:
        Rendered prompt string ready to send to the LLM.

    Raises:
        ValueError: If an unsupported version is requested.
    """
    if version == "v1":
        return USER_PROMPT_TEMPLATE_V1.format(
            query=query.strip(),
            device_context=device_context,
        )
    raise ValueError(f"Unknown prompt version: {version!r}")


def get_system_prompt(version: str = "v1") -> str:
    """
    Return the system prompt for the specified version.

    Args:
        version: Prompt version identifier.

    Returns:
        System prompt string.

    Raises:
        ValueError: If an unsupported version is requested.
    """
    if version == "v1":
        return SYSTEM_PROMPT_V1
    raise ValueError(f"Unknown prompt version: {version!r}")
