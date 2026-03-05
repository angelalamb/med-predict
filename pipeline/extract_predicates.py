"""
Extracts predicate K-numbers from previously extracted PDF text.

A predicate device is one cited as a substantial equivalence reference.
These K-numbers appear in the PDF text but are distinct from the
document's own K-number.

Output is a CSV with columns [from_k, to_k] representing
PREDICATED_ON edges ready for load_graph.load_edges().
"""

import json
import re

import pandas as pd

from config import (
    EXTRACTED_TEXT_PATH,
    DEVICES_FILTERED_PATH,
    PREDICATE_EDGES_PATH,
    get_logger,
)

logger = get_logger(__name__)


def _extract_k_numbers(text: str) -> list[str]:
    """
    Extract all K-numbers matching the FDA format (K + 6 digits) from text.

    Args:
        text: Raw extracted PDF text.

    Returns:
        List of unique uppercase K-number strings found in the text.
    """
    return list(set(re.findall(r'\bK\d{6}\b', text)))


def _build_edge_records(
    extracted: dict[str, str],
    valid_k_numbers: set[str] | None = None,
) -> list[dict]:
    """
    Build predicate edge records from extracted text.

    For each document, finds all K-numbers in the text that are not
    the document's own K-number. Optionally filters target K-numbers
    to those present in the filtered devices set.

    Args:
        extracted: Dict mapping K-number → extracted PDF text.
        valid_k_numbers: If provided, only emit edges where to_k is in
                         this set. Pass None to allow any K-number.

    Returns:
        List of {from_k, to_k} dicts.
    """
    records = []
    no_predicates = 0

    for from_k, text in extracted.items():
        from_k = from_k.upper()
        found = _extract_k_numbers(text)
        predicates = [k for k in found if k != from_k]

        if valid_k_numbers is not None:
            predicates = [k for k in predicates if k in valid_k_numbers]

        if not predicates:
            no_predicates += 1
            continue

        for to_k in predicates:
            records.append({"from_k": from_k, "to_k": to_k})

    logger.info(
        "Built %d predicate edge records (%d documents had no predicates)",
        len(records),
        no_predicates,
    )
    return records


def extract_predicate_edges(filter_to_known: bool = True) -> pd.DataFrame:
    """
    Extract PREDICATED_ON edges from extracted PDF text and save to CSV.

    Args:
        filter_to_known: If True, only emit edges where both from_k and
                         to_k are present in the filtered devices CSV.
                         Set False to capture all predicate references,
                         including devices outside the filtered set.

    Returns:
        DataFrame with columns [from_k, to_k].
    """
    if not EXTRACTED_TEXT_PATH.exists():
        raise FileNotFoundError(
            f"Extracted text not found at {EXTRACTED_TEXT_PATH}. "
            "Run extract_text.extract_text() first."
        )

    with EXTRACTED_TEXT_PATH.open("r", encoding="utf-8") as fh:
        extracted = json.load(fh)

    logger.info("Loaded extracted text for %d documents", len(extracted))

    valid_k_numbers = None
    if filter_to_known:
        if not DEVICES_FILTERED_PATH.exists():
            logger.warning(
                "Filtered devices not found at %s — "
                "emitting all predicate references without filtering",
                DEVICES_FILTERED_PATH,
            )
        else:
            df = pd.read_csv(DEVICES_FILTERED_PATH, dtype=str)
            valid_k_numbers = set(df["KNUMBER"].str.strip().str.upper().dropna())
            logger.info(
                "Filtering edges to %d known K-numbers", len(valid_k_numbers)
            )

    records = _build_edge_records(extracted, valid_k_numbers)

    edges_df = pd.DataFrame(records, columns=["from_k", "to_k"])
    edges_df.drop_duplicates(inplace=True)

    PREDICATE_EDGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(PREDICATE_EDGES_PATH, index=False)
    logger.info(
        "Saved %d unique predicate edges to %s",
        len(edges_df),
        PREDICATE_EDGES_PATH,
    )

    return edges_df