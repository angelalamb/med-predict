"""
Orchestrates the full data pipeline in sequence:
  1. Download FDA flat files
  2. Filter to neurostimulation devices
  3. Download PDFs
  4. Extract text from PDFs
  5. Parse intended use statements
  6. Generate embeddings
  7. Load graph into Neo4j

Run this script to build the full dataset from scratch.
Each step is idempotent — safe to re-run if interrupted.
"""

from config import get_logger
from pipeline.download_data import download_pdfs, download_pmn_records, download_predicate_relations, download_product_codes
from pipeline.embed import generate_embeddings
from pipeline.extract_predicates import extract_predicate_edges
from pipeline.extract_text import extract_text
from pipeline.filter_devices import filter_devices
from pipeline.load_graph import load_graph
from pipeline.parse_intended_use import parse_intended_use

logger = get_logger(__name__)


def run_pipeline() -> None:
    """
    Execute all pipeline steps in order.

    Each step logs its own progress. If a step fails it will raise,
    halting the pipeline with a clear error in the logs.
    """
    logger.info("=== MedPredict Pipeline Starting ===")

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

    # Step 2: Filter to neurostimulation devices
    logger.info("--- Step 2: Filter devices ---")
    devices_df = filter_devices()
    k_numbers = devices_df["KNUMBER"].tolist()
    logger.info("Working with %d neurostimulation K-numbers", len(k_numbers))

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
    logger.info(
        "Intended use parsed for %d documents", len(intended_use_df)
    )

    # Step 6: Generate embeddings
    logger.info("--- Step 6: Generate embeddings ---")
    generate_embeddings()

    # Step 7: Extract predicate edges from PDF text
    logger.info("--- Step 7: Extract predicate edges ---")
    extract_predicate_edges()

    # Step 8: Load into Neo4j
    logger.info("--- Step 8: Load graph into Neo4j ---")
    load_graph()

    logger.info("=== MedPredict Pipeline Complete ===")


if __name__ == "__main__":
    run_pipeline()
