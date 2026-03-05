"""
Extracts raw text from downloaded 510(k) summary PDFs using pdfplumber.
Does not attempt OCR — skips any PDF that does not yield sufficient text.

Output is a JSON file mapping K-number → extracted text.
"""

import json

import pdfplumber

from config import (
    EXTRACTED_TEXT_PATH,
    MIN_EXTRACTED_CHARS,
    PDF_DIR,
    get_logger,
)

logger = get_logger(__name__)


def _extract_text_from_pdf(pdf_path) -> str | None:
    """
    Extract all text from a PDF file using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages, or None if extraction fails
        or yields insufficient content.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            full_text = "\n".join(pages_text).strip()
    except Exception as exc:
        logger.warning("pdfplumber failed on %s: %s", pdf_path.name, exc)
        return None

    if len(full_text) < MIN_EXTRACTED_CHARS:
        logger.warning(
            "Extracted text too short (%d chars) for %s — likely scanned PDF, skipping",
            len(full_text),
            pdf_path.name,
        )
        return None

    return full_text


def _k_number_from_pdf_path(pdf_path) -> str:
    """
    Derive the K-number from a PDF filename.

    Args:
        pdf_path: Path object whose stem is the K-number (e.g. K213456).

    Returns:
        Uppercase K-number string.
    """
    return pdf_path.stem.upper()


def _load_existing_extractions() -> dict[str, str]:
    """
    Load any previously extracted text from disk to support incremental runs.

    Returns:
        Dict mapping K-number → extracted text, or empty dict if none exists.
    """
    if not EXTRACTED_TEXT_PATH.exists():
        return {}

    try:
        with EXTRACTED_TEXT_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info(
            "Loaded %d existing extractions from %s",
            len(data),
            EXTRACTED_TEXT_PATH,
        )
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not load existing extractions from %s: %s",
            EXTRACTED_TEXT_PATH,
            exc,
        )
        return {}


def _save_extractions(data: dict[str, str]) -> None:
    """
    Persist the extractions dict to disk.

    Args:
        data: Dict mapping K-number → extracted text.
    """
    with EXTRACTED_TEXT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info(
        "Saved %d extractions to %s",
        len(data),
        EXTRACTED_TEXT_PATH,
    )


def _get_pdf_paths(k_numbers: list[str] | None) -> list:
    """
    Collect PDF paths to process, optionally filtered by K-number list.

    Args:
        k_numbers: If provided, restrict to these K-numbers.
                   If None, process all PDFs in PDF_DIR.

    Returns:
        List of Path objects for PDFs to process.
    """
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))

    if k_numbers is None:
        return all_pdfs

    target_set = {k.upper() for k in k_numbers}
    return [p for p in all_pdfs if p.stem.upper() in target_set]


def extract_text(k_numbers: list[str] | None = None) -> dict[str, str]:
    """
    Extract text from all downloaded PDFs (or a filtered subset).

    Skips PDFs that have already been extracted (incremental/idempotent).
    Skips PDFs where pdfplumber returns insufficient text.

    Args:
        k_numbers: Optional list of K-numbers to restrict processing to.
                   If None, all PDFs in PDF_DIR are processed.

    Returns:
        Dict mapping K-number → extracted text for all successfully
        extracted documents (including previously cached results).
    """
    pdf_paths = _get_pdf_paths(k_numbers)
    total = len(pdf_paths)
    logger.info("Found %d PDFs to process", total)

    existing = _load_existing_extractions()
    results = dict(existing)

    success_count = 0
    skipped_existing = 0
    failed_count = 0

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        k_number = _k_number_from_pdf_path(pdf_path)

        if k_number in results:
            skipped_existing += 1
            logger.debug("Already extracted %s — skipping", k_number)
            continue

        text = _extract_text_from_pdf(pdf_path)

        if text is not None:
            results[k_number] = text
            success_count += 1
            logger.debug(
                "Extracted %d chars from %s", len(text), k_number
            )
        else:
            failed_count += 1

        if idx % 100 == 0 or idx == total:
            logger.info(
                "Extraction progress: %d/%d — "
                "success=%d, skipped_existing=%d, failed=%d",
                idx,
                total,
                success_count,
                skipped_existing,
                failed_count,
            )

    _save_extractions(results)

    logger.info(
        "Extraction complete: %d total in cache, "
        "new_success=%d, skipped_existing=%d, failed=%d",
        len(results),
        success_count,
        skipped_existing,
        failed_count,
    )

    return results
