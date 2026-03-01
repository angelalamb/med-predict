"""
Central configuration for MedPredict.
All constants, paths, and settings live here.
Modules import from this file rather than hardcoding values.
"""

import logging
import logging.handlers
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = RAW_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOG_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for _dir in [RAW_DIR, PDF_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FDA Data Sources
# ---------------------------------------------------------------------------

FDA_510K_URL = (
    "https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip"
)
FDA_PRODUCT_CODE_URL = (
    "https://www.accessdata.fda.gov/premarket/ftparea/foiclass.zip"
)
FDA_PDF_BASE_URL = (
    "https://www.accessdata.fda.gov/cdrh_docs/pdf"
)
FDA_PREDICATE_URL = (
    "https://www.accessdata.fda.gov/premarket/ftparea/pmnrelat.zip"
)

PMN_RAW_PATH = RAW_DIR / "pmn_records.csv"
PRODUCT_CODE_RAW_PATH = RAW_DIR / "product_codes.csv"
PMN_RELAT_PATH = RAW_DIR / "pmn_relat.csv"
DEVICES_FILTERED_PATH = PROCESSED_DIR / "devices_filtered.csv"
INTENDED_USE_PATH = PROCESSED_DIR / "intended_use.csv"
PDF_MANIFEST_PATH = PROCESSED_DIR / "pdf_manifest.csv"
EXTRACTED_TEXT_PATH = PROCESSED_DIR / "extracted_text.json"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_DIR / "embeddings_cache.pkl"
PREDICATE_EDGES_PATH = PROCESSED_DIR / "predicate_edges.csv"

# ---------------------------------------------------------------------------
# Neurostimulation Product Codes
# ---------------------------------------------------------------------------

NEUROSTIMULATION_PRODUCT_CODES = [
    "GZP",  # Implantable spinal cord stimulator
    "LLD",  # Deep brain stimulator
    "NPN",  # Neurostimulator, implantable
    "QFN",  # Implantable pulse generator for pain
    "MRX",  # Transcutaneous electrical nerve stimulator
    "IYO",  # Vagus nerve stimulator
    "OZO",  # Sacral nerve stimulator
    "PZI",  # Peripheral nerve stimulator
]

# Only include cleared submissions
CLEARED_DECISION_CODES = ["SESE", "SE"]

# Only include submissions from this year onwards (avoids messy old PDFs)
MIN_SUBMISSION_YEAR = 2005

# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

# Minimum characters for extracted text to be considered valid
MIN_EXTRACTED_CHARS = 100

# Seconds to wait between PDF download requests
PDF_DOWNLOAD_DELAY = 1.5

# Intended use section header variants to search for
INTENDED_USE_HEADERS = [
    "Indications for Use",
    "Indications For Use",
    "INDICATIONS FOR USE",
    "Intended Use",
    "INTENDED USE",
    "Indications:",
    "INDICATIONS:",
    "Indication for Use",
]

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIMENSION = 768

# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

NEO4J_BATCH_SIZE = 500  # nodes/edges per transaction

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

SEMANTIC_TOP_K = 5          # Number of candidates from vector search
GRAPH_TRAVERSAL_DEPTH = 2   # Hops to traverse from seed nodes

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "medpredict.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 5


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with both a console handler and a
    rotating file handler.  All modules call this function rather than
    instantiating their own loggers.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
