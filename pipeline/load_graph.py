"""
Loads filtered device records and embeddings into Neo4j AuraDB.

Two passes:
  1. Create Device nodes with structured attributes and intended use text
  2. Create PREDICATED_ON edges from the PREDICATENUMBER field
  3. Update nodes with embedding vectors

All writes are batched for performance.
"""

import ast
import re

import pandas as pd
from neo4j import GraphDatabase, basic_auth

from config import (
    DEVICES_FILTERED_PATH,
    EMBEDDINGS_CACHE_PATH,
    INTENDED_USE_PATH,
    NEO4J_BATCH_SIZE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    PREDICATE_EDGES_PATH,
    get_logger,
)
from pipeline.embed import load_cached_embeddings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cypher statements
# ---------------------------------------------------------------------------

_CREATE_CONSTRAINT = """
CREATE CONSTRAINT device_k_number_unique IF NOT EXISTS
FOR (d:Device) REQUIRE d.k_number IS UNIQUE
"""

_CREATE_VECTOR_INDEX = """
CREATE VECTOR INDEX device_embedding_index IF NOT EXISTS
FOR (d:Device) ON (d.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 768,
  `vector.similarity_function`: 'cosine'
}}
"""

_MERGE_DEVICE_NODE = """
UNWIND $batch AS row
MERGE (d:Device {k_number: row.k_number})
SET d.device_name    = row.device_name,
    d.applicant      = row.applicant,
    d.product_code   = row.product_code,
    d.decision_date  = row.decision_date,
    d.intended_use   = row.intended_use
"""

_CREATE_PREDICATED_ON_EDGE = """
UNWIND $batch AS row
MATCH (a:Device {k_number: row.from_k})
MATCH (b:Device {k_number: row.to_k})
MERGE (a)-[:PREDICATED_ON]->(b)
"""

_SET_EMBEDDING = """
UNWIND $batch AS row
MATCH (d:Device {k_number: row.k_number})
SET d.embedding = row.embedding
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_driver():
    """
    Create and return a Neo4j driver instance.

    Returns:
        neo4j.GraphDatabase.Driver

    Raises:
        RuntimeError: If connection cannot be established.
    """
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=basic_auth(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s", NEO4J_URI)
        return driver
    except Exception as exc:
        logger.error("Failed to connect to Neo4j: %s", exc)
        raise RuntimeError(f"Neo4j connection failed: {exc}") from exc


def _run_batched(driver, cypher: str, batch: list[dict]) -> int:
    """
    Execute a Cypher write statement over a list of parameter dicts in
    one transaction.

    Args:
        driver: Active Neo4j driver.
        cypher: Parameterised Cypher statement using $batch.
        batch: List of parameter dicts.

    Returns:
        Number of items in the batch (for logging).
    """
    with driver.session() as session:
        session.run(cypher, batch=batch)
    return len(batch)


def _run_in_batches(
    driver,
    cypher: str,
    records: list[dict],
    batch_size: int,
    label: str,
) -> None:
    """
    Split records into batches and run a Cypher statement for each.

    Args:
        driver: Active Neo4j driver.
        cypher: Parameterised Cypher statement.
        records: Full list of parameter dicts.
        batch_size: Number of records per transaction.
        label: Human-readable label for logging.
    """
    total = len(records)
    created = 0

    for batch_start in range(0, total, batch_size):
        batch = records[batch_start : batch_start + batch_size]
        _run_batched(driver, cypher, batch)
        created += len(batch)
        logger.info("%s: %d/%d written", label, created, total)


# ---------------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------------


def create_schema(driver) -> None:
    """
    Create uniqueness constraint and vector index in Neo4j.
    Uses IF NOT EXISTS so this is safe to re-run.

    Args:
        driver: Active Neo4j driver.
    """
    with driver.session() as session:
        session.run(_CREATE_CONSTRAINT)
        logger.info("Device uniqueness constraint ensured")
        session.run(_CREATE_VECTOR_INDEX)
        logger.info("Device embedding vector index ensured")


# ---------------------------------------------------------------------------
# Node loading
# ---------------------------------------------------------------------------


def _load_devices_df() -> pd.DataFrame:
    """
    Load the filtered devices CSV.

    Returns:
        DataFrame of filtered device records.

    Raises:
        FileNotFoundError: If the filtered CSV does not exist.
    """
    if not DEVICES_FILTERED_PATH.exists():
        raise FileNotFoundError(
            f"Filtered devices not found at {DEVICES_FILTERED_PATH}. "
            "Run filter_devices.filter_devices() first."
        )
    df = pd.read_csv(DEVICES_FILTERED_PATH, dtype=str)
    logger.info("Loaded %d filtered device records", len(df))
    return df


def _load_intended_use_map() -> dict[str, str]:
    """
    Load the intended use CSV into a K-number → text lookup dict.

    Returns:
        Dict, or empty dict if file does not exist.
    """
    if not INTENDED_USE_PATH.exists():
        logger.warning(
            "Intended use file not found at %s — nodes will be loaded "
            "without intended use text",
            INTENDED_USE_PATH,
        )
        return {}

    df = pd.read_csv(INTENDED_USE_PATH, dtype=str).dropna(
        subset=["k_number", "intended_use_text"]
    )
    mapping = dict(zip(df["k_number"], df["intended_use_text"]))
    logger.info("Loaded %d intended use statements", len(mapping))
    return mapping


def _build_node_records(
    devices_df: pd.DataFrame,
    intended_use_map: dict[str, str],
) -> list[dict]:
    """
    Build the list of parameter dicts for node creation.

    Args:
        devices_df: Filtered devices DataFrame.
        intended_use_map: K-number → intended use text.

    Returns:
        List of dicts ready for the Cypher UNWIND statement.
    """
    records = []
    for _, row in devices_df.iterrows():
        k = row.get("KNUMBER", "").strip().upper()
        if not k:
            continue
        records.append(
            {
                "k_number": k,
                "device_name": row.get("DEVICENAME", ""),
                "applicant": row.get("APPLICANT", ""),
                "product_code": row.get("PRODUCTCODE", ""),
                "decision_date": str(row.get("DECISIONDATE", "")),
                "intended_use": intended_use_map.get(k, ""),
            }
        )
    return records


def load_nodes(driver) -> None:
    """
    Create or update Device nodes in Neo4j from the filtered devices CSV.

    Args:
        driver: Active Neo4j driver.
    """
    devices_df = _load_devices_df()
    intended_use_map = _load_intended_use_map()
    records = _build_node_records(devices_df, intended_use_map)

    logger.info("Loading %d Device nodes into Neo4j", len(records))
    _run_in_batches(driver, _MERGE_DEVICE_NODE, records, NEO4J_BATCH_SIZE, "Nodes")
    logger.info("Node loading complete")


# ---------------------------------------------------------------------------
# Edge loading
# ---------------------------------------------------------------------------


def _parse_predicate_numbers(raw: str) -> list[str]:
    """
    Parse the PREDICATENUMBER field, which may contain one or more
    K-numbers separated by semicolons, commas, or spaces.

    Args:
        raw: Raw string value of the PREDICATENUMBER field.

    Returns:
        List of cleaned, uppercase K-number strings.
    """
    if not raw or pd.isna(raw):
        return []

    # Split on common separators
    parts = re.split(r"[;,\s]+", str(raw).strip())
    k_numbers = []
    for part in parts:
        cleaned = part.strip().upper()
        # Validate basic K-number format: K followed by digits
        if re.match(r"^K\d+$", cleaned):
            k_numbers.append(cleaned)
    return k_numbers


def _build_edge_records(devices_df: pd.DataFrame) -> list[dict]:
    """
    Load predicate edge records from the extracted predicate edges CSV.

    Returns:
        List of {from_k, to_k} dicts for the Cypher UNWIND statement.
    """
    if not PREDICATE_EDGES_PATH.exists():
        logger.warning(
            "Predicate edges not found at %s — no edges will be loaded. "
            "Run extract_predicates.extract_predicate_edges() first.",
            PREDICATE_EDGES_PATH,
        )
        return []

    df = pd.read_csv(PREDICATE_EDGES_PATH, dtype=str).dropna()
    records = [
        {"from_k": row["from_k"].strip().upper(), "to_k": row["to_k"].strip().upper()}
        for _, row in df.iterrows()
    ]
    logger.info("Loaded %d predicate edge records from CSV", len(records))
    return records


def load_edges(driver) -> None:
    """
    Create PREDICATED_ON edges in Neo4j from the predicate edges CSV.

    Only creates edges where both the source and target Device nodes
    already exist in the graph (MATCH semantics, not MERGE on missing nodes).

    Args:
        driver: Active Neo4j driver.
    """
    records = _build_edge_records(devices_df=None)

    logger.info("Loading %d PREDICATED_ON edges into Neo4j", len(records))
    _run_in_batches(
        driver, _CREATE_PREDICATED_ON_EDGE, records, NEO4J_BATCH_SIZE, "Edges"
    )
    logger.info("Edge loading complete")


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------


def _build_embedding_records(
    cache: dict[str, list[float]],
) -> list[dict]:
    """
    Build parameter dicts for the embedding SET statement.

    Args:
        cache: K-number → embedding list.

    Returns:
        List of {k_number, embedding} dicts.
    """
    return [
        {"k_number": k_number, "embedding": embedding}
        for k_number, embedding in cache.items()
    ]


def load_embeddings(driver) -> None:
    """
    Update Device nodes in Neo4j with their embedding vectors.

    Args:
        driver: Active Neo4j driver.
    """
    if not EMBEDDINGS_CACHE_PATH.exists():
        logger.warning(
            "Embeddings cache not found at %s — skipping embedding load",
            EMBEDDINGS_CACHE_PATH,
        )
        return

    cache = load_cached_embeddings()
    records = _build_embedding_records(cache)

    logger.info("Loading embeddings for %d devices into Neo4j", len(records))
    _run_in_batches(
        driver, _SET_EMBEDDING, records, NEO4J_BATCH_SIZE, "Embeddings"
    )
    logger.info("Embedding load complete")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def load_graph() -> None:
    """
    Full graph loading pipeline: schema → nodes → edges → embeddings.

    Creates a single driver, runs all three loading steps, then closes
    the connection.
    """
    driver = _get_driver()

    try:
        create_schema(driver)
        load_nodes(driver)
        load_edges(driver)
        load_embeddings(driver)
        logger.info("Graph loading pipeline complete")
    finally:
        driver.close()
        logger.info("Neo4j driver closed")
