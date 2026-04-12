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

for _dir in [RAW_DIR, PDF_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FDA Data Sources
# ---------------------------------------------------------------------------

FDA_510K_URL = "https://www.accessdata.fda.gov/premarket/ftparea/pmn96cur.zip"
FDA_PRODUCT_CODE_URL = "https://www.accessdata.fda.gov/premarket/ftparea/foiclass.zip"
FDA_PDF_BASE_URL = "https://www.accessdata.fda.gov/cdrh_docs/pdf"
FDA_PREDICATE_URL = "https://www.accessdata.fda.gov/premarket/ftparea/pmnrelat.zip"

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
# Ultrasound Product Codes
# ---------------------------------------------------------------------------

ULTRASOUND_PRODUCT_CODES = [
    "IYO",  # System, Imaging, Pulsed Echo, Ultrasonic
    "IYN",  # System, Imaging, Pulsed Doppler, Ultrasonic
    "ITX",  # Transducer, Ultrasonic, Diagnostic
]

CLEARED_DECISION_CODES = ["SESE", "SE"]
MIN_SUBMISSION_YEAR = 2005

# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

MIN_EXTRACTED_CHARS = 100
PDF_DOWNLOAD_DELAY = 1.5

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

SEMANTIC_TOP_K = 5        # Number of candidates from vector search
GRAPH_TRAVERSAL_DEPTH = 2  # Hops to traverse from seed nodes

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Claude Sonnet token pricing (USD per token)
CLAUDE_INPUT_TOKEN_COST = 0.000003
CLAUDE_OUTPUT_TOKEN_COST = 0.000015

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

API_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# MLflow / Databricks Tracking
# ---------------------------------------------------------------------------

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")  # never logged
MLFLOW_SAMPLE_RATE = float(os.getenv("MLFLOW_SAMPLE_RATE", "0.1"))
JUDGE_MODEL = os.getenv("JUDGE_MODEL", LLM_MODEL)
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

MLFLOW_EXPERIMENT_PIPELINE = "/medpredict/pipeline_runs"
MLFLOW_EXPERIMENT_QUERY = "/medpredict/query_metrics"
MLFLOW_EXPERIMENT_EVAL = "/medpredict/llm_judge_eval"

# LLM judge prompt constraints
JUDGE_MAX_DEVICES = 20          # max devices included in judge prompt
JUDGE_MAX_ANALYSIS_CHARS = 3000  # cap analysis length sent to judge
JUDGE_MAX_TOKENS = 512          # max tokens for judge response (reasoning + JSON)

# Weekly evaluation
JUDGE_SCORE_PASS_THRESHOLD = 3.5  # weekly avg below this triggers a warning
JUDGE_LOOKBACK_DAYS = 7

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOG_DIR / "medpredict.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 5


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with a console handler and a rotating file
    handler. All modules call this rather than instantiating their own loggers.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def validate() -> None:
    """
    Validate that all required environment variables are set.

    Raises:
        ValueError: If any required variable is missing.
    """
    required = {
        "NEO4J_URI": NEO4J_URI,
        "NEO4J_PASSWORD": NEO4J_PASSWORD,
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    }

    missing = [var for var, val in required.items() if not val]

    if missing:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "Set these in your .env file or deployment environment variables."
        )
