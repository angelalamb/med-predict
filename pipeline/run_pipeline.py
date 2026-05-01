"""
Orchestrates the full data pipeline in sequence:
  1. Download FDA flat files
  2. Filter to devices for the specified category
  3. Download PDFs
  4. Extract text from PDFs
  5. Parse intended use statements
  6. Generate embeddings
  7. Extract predicate edges
  8. Load graph into Neo4j

Run for a single category:
    python -m pipeline.run_pipeline --category ultrasound

Run all categories in sequence:
    python -m pipeline.run_pipeline --all

Each step is idempotent — safe to re-run if interrupted.
The graph is additive: running multiple categories populates Neo4j
without overwriting existing nodes.
"""

import argparse
import time
from datetime import datetime, timezone
import config
from config import DEVICE_CATEGORIES, get_logger
from pipeline.download_data import download_pdfs, download_pmn_records, download_predicate_relations, download_product_codes
from pipeline.embed import generate_embeddings
from pipeline.extract_predicates import extract_predicate_edges
from pipeline.extract_text import extract_text
from pipeline.filter_devices import filter_devices
from pipeline.load_graph import load_graph
from pipeline.parse_intended_use import parse_intended_use
from tracking import tracker

logger = get_logger(__name__)


def run_pipeline(category: str) -> None:
    """
    Execute all pipeline steps for a single device category.

    Args:
        category: Key from DEVICE_CATEGORIES (e.g. "ultrasound").

    Raises:
        ValueError: If category is not defined in DEVICE_CATEGORIES.
    """
    if category not in DEVICE_CATEGORIES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories: {list(DEVICE_CATEGORIES.keys())}"
        )

    cat = DEVICE_CATEGORIES[category]
    logger.info("=== MedPredict Pipeline Starting — category: %s (%s) ===", category, cat["label"])

    run_name = f"pipeline_{category}_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t_start = time.perf_counter()

    with tracker.pipeline_run(run_name=run_name) as run:
        # Step 1: Download flat files
        logger.info("--- Step 1: Download FDA flat files ---")
        pmn_ok = download_pmn_records()
        pc_ok = download_product_codes()
        if not pmn_ok or not pc_ok:
            logger.error("Flat file download failed — aborting pipeline")
            return

        relat_ok = download_predicate_relations()
        if not relat_ok:
            logger.warning(
                "Predicate relations file unavailable — graph will have nodes "
                "but no predicate edges. Edges will be extracted from PDFs later."
            )

        # Step 2: Filter devices for this category
        logger.info("--- Step 2: Filter devices (category: %s) ---", category)
        devices_df = filter_devices(category)
        k_numbers = devices_df["KNUMBER"].tolist()
        logger.info("Working with %d K-numbers for category '%s'", len(k_numbers), category)

        # Step 3: Download PDFs
        logger.info("--- Step 3: Download PDFs ---")
        download_pdfs(k_numbers)

        # Step 4: Extract text
        logger.info("--- Step 4: Extract text from PDFs ---")
        extracted = extract_text(k_numbers)
        logger.info("Text extracted for %d documents", len(extracted))

        # Step 5: Parse intended use
        logger.info("--- Step 5: Parse intended use statements ---")
        intended_use_df = parse_intended_use(extracted)
        logger.info("Intended use parsed for %d documents", len(intended_use_df))

        # Step 6: Generate embeddings
        logger.info("--- Step 6: Generate embeddings ---")
        generate_embeddings()

        # Step 7: Extract predicate edges
        logger.info("--- Step 7: Extract predicate edges ---")
        extract_predicate_edges()

        # Step 8: Load into Neo4j
        logger.info("--- Step 8: Load graph into Neo4j ---")
        load_graph(category)

        extracted_count = len(extracted)
        intended_use_count = len(intended_use_df)
        k_numbers_count = len(k_numbers)

        tracker.log_pipeline_metrics(
            run,
            {
                "k_numbers_count": float(k_numbers_count),
                "extracted_count": float(extracted_count),
                "intended_use_count": float(intended_use_count),
                "text_extraction_rate": extracted_count / k_numbers_count if k_numbers_count else 0.0,
                "intended_use_rate": intended_use_count / extracted_count if extracted_count else 0.0,
                "total_duration_s": time.perf_counter() - t_start,
            },
            category=category,
        )

    logger.info("=== MedPredict Pipeline Complete — category: %s ===", category)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MedPredict data pipeline.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--category",
        choices=list(DEVICE_CATEGORIES.keys()),
        help="Run pipeline for a single device category.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Run pipeline for all device categories in sequence.",
    )
    args = parser.parse_args()

    config.validate()

    if args.all:
        for category in DEVICE_CATEGORIES:
            run_pipeline(category)
    else:
        run_pipeline(args.category)


if __name__ == "__main__":
    main()
