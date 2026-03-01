"""
pipeline/parse_intended_use.py

Parses intended use / indications for use statements from raw extracted
PDF text.  Uses header-based section detection followed by text cleanup.

Writes results to INTENDED_USE_PATH as a CSV.
"""

import re

import pandas as pd

from config import (
    INTENDED_USE_HEADERS,
    INTENDED_USE_PATH,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Section boundary detection
# ---------------------------------------------------------------------------

# Regex that matches the start of a new major section (numbered or headed).
# Used to find where the intended use section ends.
_NEXT_SECTION_PATTERN = re.compile(
    r"^\s*(?:\d+[\.\)]\s+[A-Z]|[A-Z]{2,}[\s:]+)",
    re.MULTILINE,
)


def _build_header_pattern(headers: list[str]) -> re.Pattern:
    """
    Compile a regex that matches any of the known intended use headers.

    Args:
        headers: List of header string variants.

    Returns:
        Compiled regex pattern.
    """
    escaped = [re.escape(h) for h in headers]
    pattern = r"(?:" + "|".join(escaped) + r")\s*[:\-]?\s*"
    return re.compile(pattern, re.IGNORECASE)


_HEADER_PATTERN = _build_header_pattern(INTENDED_USE_HEADERS)


def _find_header_position(text: str) -> int | None:
    """
    Find the character position immediately after the intended use header.

    Args:
        text: Full extracted document text.

    Returns:
        Character index where the intended use content starts,
        or None if no header found.
    """
    match = _HEADER_PATTERN.search(text)
    if match is None:
        return None
    return match.end()


def _extract_section_text(text: str, start: int) -> str:
    """
    Extract the text from start position until the next major section header.

    Args:
        text: Full document text.
        start: Character index where the intended use content begins.

    Returns:
        Raw section text, potentially spanning multiple lines.
    """
    remaining = text[start:]

    # Find the next section header after our start position
    next_section = _NEXT_SECTION_PATTERN.search(remaining)
    if next_section and next_section.start() > 20:
        # Only truncate if the next section is not immediately adjacent
        return remaining[: next_section.start()]

    # Fall back: take up to 1000 characters (intended use is never that long)
    return remaining[:1000]


def _clean_section_text(raw: str) -> str:
    """
    Clean extracted section text: collapse whitespace, remove page artifacts.

    Args:
        raw: Raw text extracted for the intended use section.

    Returns:
        Cleaned string.
    """
    # Replace various whitespace sequences with single space
    text = re.sub(r"\s+", " ", raw)
    # Remove common PDF artifacts: lone numbers (page numbers), form feeds
    text = re.sub(r"\s+\d{1,3}\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def _is_valid_intended_use(text: str) -> bool:
    """
    Basic validity check: is the extracted text plausibly an intended use?

    Args:
        text: Cleaned intended use text.

    Returns:
        True if text meets minimum quality criteria.
    """
    if len(text) < 20:
        return False
    # Should contain at least one verb-like word common in intended use
    keywords = ["intended", "indicated", "designed", "used", "device", "patient"]
    lowered = text.lower()
    return any(kw in lowered for kw in keywords)


def _parse_single_document(k_number: str, text: str) -> str | None:
    """
    Attempt to extract the intended use statement from one document's text.

    Args:
        k_number: K-number (used for logging only).
        text: Full extracted text of the 510(k) summary.

    Returns:
        Cleaned intended use string, or None if not found / invalid.
    """
    start = _find_header_position(text)

    if start is None:
        logger.warning("No intended use header found in %s", k_number)
        return None

    raw_section = _extract_section_text(text, start)
    cleaned = _clean_section_text(raw_section)

    if not _is_valid_intended_use(cleaned):
        logger.warning(
            "Extracted text for %s failed validity check: %r",
            k_number,
            cleaned[:80],
        )
        return None

    logger.debug(
        "Parsed intended use for %s (%d chars): %r",
        k_number,
        len(cleaned),
        cleaned[:80],
    )
    return cleaned


def parse_intended_use(
    extracted_texts: dict[str, str],
) -> pd.DataFrame:
    """
    Parse intended use statements from a dict of extracted PDF texts.

    Args:
        extracted_texts: Dict mapping K-number → full extracted text.

    Returns:
        DataFrame with columns:
          - k_number
          - intended_use_text
          - char_count
        Only rows where parsing succeeded are included.
        Also written to INTENDED_USE_PATH.
    """
    total = len(extracted_texts)
    logger.info(
        "Parsing intended use statements from %d extracted documents", total
    )

    records = []
    success_count = 0
    failed_count = 0

    for k_number, text in extracted_texts.items():
        intended_use = _parse_single_document(k_number, text)

        if intended_use is not None:
            records.append(
                {
                    "k_number": k_number,
                    "intended_use_text": intended_use,
                    "char_count": len(intended_use),
                }
            )
            success_count += 1
        else:
            failed_count += 1

    df = pd.DataFrame(records)

    if not df.empty:
        df.to_csv(INTENDED_USE_PATH, index=False)
        logger.info(
            "Parsing complete: %d/%d succeeded (%.1f%%). "
            "Average char count: %.0f. Written to %s",
            success_count,
            total,
            100 * success_count / total if total else 0,
            df["char_count"].mean(),
            INTENDED_USE_PATH,
        )
    else:
        logger.error(
            "No intended use statements were successfully parsed from %d documents",
            total,
        )

    return df
