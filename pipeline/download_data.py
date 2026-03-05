"""
Handles two download tasks:
  1. FDA 510(k) bulk flat files (structured records and product codes)
  2. Individual 510(k) summary PDFs for filtered devices

Each function does one thing and logs its outcome.
"""

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import (
    FDA_510K_URL,
    FDA_PDF_BASE_URL,
    FDA_PREDICATE_URL,
    FDA_PRODUCT_CODE_URL,
    PDF_DIR,
    PDF_DOWNLOAD_DELAY,
    PDF_MANIFEST_PATH,
    PMN_RAW_PATH,
    PMN_RELAT_PATH,
    PRODUCT_CODE_RAW_PATH,
    get_logger,
)

logger = get_logger(__name__)

# FDA abuse detection blocks requests without browser-like headers.
# These headers are added to every request made by this module.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Flat file downloads
# ---------------------------------------------------------------------------


def _download_zip_to_csv(url: str, output_path: Path) -> bool:
    """
    Download a zip archive from url, extract the first file inside it,
    and write it as a CSV to output_path.

    Args:
        url: URL of the zip archive.
        output_path: Destination path for the extracted CSV.

    Returns:
        True on success, False on failure.
    """
    logger.info("Downloading %s", url)
    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("Failed to download %s: %s", url, exc)
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            if not names:
                logger.error("Zip archive from %s is empty", url)
                return False
            target = names[0]
            logger.debug("Extracting %s from archive", target)
            raw_bytes = zf.read(target)

        output_path.write_bytes(raw_bytes)
        logger.info(
            "Saved %s (%.1f MB) to %s",
            target,
            len(raw_bytes) / 1_048_576,
            output_path,
        )
        return True

    except (zipfile.BadZipFile, KeyError) as exc:
        logger.error("Failed to extract zip from %s: %s", url, exc)
        return False


def download_pmn_records() -> bool:
    """
    Download the FDA 510(k) premarket notification bulk flat file.

    Returns:
        True if the file was downloaded and saved successfully.
    """
    if PMN_RAW_PATH.exists():
        logger.info(
            "PMN records already exist at %s — skipping download",
            PMN_RAW_PATH,
        )
        return True

    return _download_zip_to_csv(FDA_510K_URL, PMN_RAW_PATH)


def download_product_codes() -> bool:
    """
    Download the FDA product classification flat file.

    Returns:
        True if the file was downloaded and saved successfully.
    """
    if PRODUCT_CODE_RAW_PATH.exists():
        logger.info(
            "Product codes already exist at %s — skipping download",
            PRODUCT_CODE_RAW_PATH,
        )
        return True

    return _download_zip_to_csv(FDA_PRODUCT_CODE_URL, PRODUCT_CODE_RAW_PATH)



def download_predicate_relations() -> bool:
    """
    Download the FDA 510(k) predicate relationship file.

    This is a separate flat file from the main PMN records that contains
    the predicate K-number for each submission.

    Returns:
        True if the file was downloaded or already exists.
        False if the download fails (non-fatal — pipeline continues).
    """
    if PMN_RELAT_PATH.exists():
        logger.info(
            "Predicate relations already exist at %s — skipping download",
            PMN_RELAT_PATH,
        )
        return True

    url = FDA_PREDICATE_URL
    logger.info("Downloading %s", url)
    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=60)
        if response.status_code == 404:
            logger.warning(
                "Predicate relations file not found at %s (404). "
                "The pipeline will continue without predicate edges. "
                "These can be extracted from PDFs in a later step.",
                url,
            )
            return False
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Could not download predicate relations: %s", exc)
        return False

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = zf.namelist()
            if not names:
                logger.warning("Predicate relations zip archive is empty")
                return False
            raw_bytes = zf.read(names[0])
        PMN_RELAT_PATH.write_bytes(raw_bytes)
        logger.info(
            "Saved predicate relations (%.1f MB) to %s",
            len(raw_bytes) / 1_048_576,
            PMN_RELAT_PATH,
        )
        return True
    except (zipfile.BadZipFile, KeyError) as exc:
        logger.warning("Failed to extract predicate relations zip: %s", exc)
        return False

# ---------------------------------------------------------------------------
# PDF downloads
# ---------------------------------------------------------------------------


def _build_pdf_url(k_number: str) -> str:
    """
    Construct the FDA URL for a given 510(k) summary PDF.

    The FDA stores PDFs under subdirectories named by the first part of
    the K-number, e.g. K213456 → /cdrh_docs/pdf21/K213456.pdf

    Args:
        k_number: The 510(k) K-number, e.g. 'K213456'.

    Returns:
        Full URL string.
    """
    # Extract numeric suffix — K213456 → '21'
    numeric = k_number.lstrip("Kk")
    subdir = numeric[:2]
    filename = f"{k_number.upper()}.pdf"
    return f"{FDA_PDF_BASE_URL}{subdir}/{filename}"


def _pdf_already_downloaded(k_number: str) -> bool:
    """
    Check whether the PDF for a given K-number is already on disk.

    Args:
        k_number: The 510(k) K-number.

    Returns:
        True if file exists and has non-zero size.
    """
    path = PDF_DIR / f"{k_number.upper()}.pdf"
    return path.exists() and path.stat().st_size > 0


def _download_single_pdf(k_number: str) -> str:
    """
    Attempt to download the summary PDF for one K-number.

    Args:
        k_number: The 510(k) K-number.

    Returns:
        'success', 'skipped', or 'failed'.
    """
    if _pdf_already_downloaded(k_number):
        logger.debug("PDF already exists for %s — skipping", k_number)
        return "skipped"

    url = _build_pdf_url(k_number)
    dest = PDF_DIR / f"{k_number.upper()}.pdf"

    try:
        response = requests.get(url, headers=_REQUEST_HEADERS, timeout=30)
        if response.status_code == 404:
            logger.warning("PDF not found (404) for %s at %s", k_number, url)
            return "failed"
        response.raise_for_status()
        dest.write_bytes(response.content)
        logger.debug("Downloaded PDF for %s (%d bytes)", k_number, len(response.content))
        return "success"

    except requests.RequestException as exc:
        logger.warning("Failed to download PDF for %s: %s", k_number, exc)
        return "failed"


def download_pdfs(k_numbers: list[str]) -> pd.DataFrame:
    """
    Download summary PDFs for a list of K-numbers, respecting rate limits.
    Writes a manifest CSV recording the outcome for each K-number.
    Skips K-numbers whose PDFs are already on disk.

    Args:
        k_numbers: List of 510(k) K-numbers to download.

    Returns:
        DataFrame with columns [k_number, status] summarising outcomes.
    """
    total = len(k_numbers)
    logger.info("Starting PDF downloads for %d K-numbers", total)

    results = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for idx, k_number in enumerate(k_numbers, start=1):
        status = _download_single_pdf(k_number)
        results.append({"k_number": k_number, "status": status})

        if status == "success":
            success_count += 1
            time.sleep(PDF_DOWNLOAD_DELAY)
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

        if idx % 50 == 0 or idx == total:
            logger.info(
                "Progress: %d/%d — success=%d, skipped=%d, failed=%d",
                idx,
                total,
                success_count,
                skipped_count,
                failed_count,
            )

    manifest = pd.DataFrame(results)
    manifest.to_csv(PDF_MANIFEST_PATH, index=False)
    logger.info(
        "PDF download complete. success=%d, skipped=%d, failed=%d. "
        "Manifest saved to %s",
        success_count,
        skipped_count,
        failed_count,
        PDF_MANIFEST_PATH,
    )

    return manifest
