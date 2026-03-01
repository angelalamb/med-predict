"""
graph/schema.py

Defines and creates Neo4j schema elements:
  - Uniqueness constraint on Device.k_number
  - Vector index on Device.embedding for similarity search

All statements use IF NOT EXISTS so this module is safe to call on
an already-initialised database.
"""

from graph.connection import get_session
from config import EMBEDDING_DIMENSION, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Cypher DDL statements
# ---------------------------------------------------------------------------

_CONSTRAINT_DEVICE_K_NUMBER = """
CREATE CONSTRAINT device_k_number_unique IF NOT EXISTS
FOR (d:Device) REQUIRE d.k_number IS UNIQUE
"""

_INDEX_DEVICE_VECTOR = f"""
CREATE VECTOR INDEX device_embedding_index IF NOT EXISTS
FOR (d:Device) ON (d.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {EMBEDDING_DIMENSION},
  `vector.similarity_function`: 'cosine'
}}}}
"""

_INDEX_DEVICE_PRODUCT_CODE = """
CREATE INDEX device_product_code_index IF NOT EXISTS
FOR (d:Device) ON (d.product_code)
"""

_INDEX_DEVICE_DECISION_DATE = """
CREATE INDEX device_decision_date_index IF NOT EXISTS
FOR (d:Device) ON (d.decision_date)
"""


# ---------------------------------------------------------------------------
# Individual schema creation functions
# ---------------------------------------------------------------------------


def create_uniqueness_constraint() -> None:
    """
    Create a uniqueness constraint on Device.k_number.

    Ensures no two Device nodes share the same K-number and provides
    an implicit index for fast K-number lookups.
    """
    with get_session() as session:
        session.run(_CONSTRAINT_DEVICE_K_NUMBER)
    logger.info("Uniqueness constraint on Device.k_number ensured")


def create_vector_index() -> None:
    """
    Create the vector index on Device.embedding for cosine similarity search.

    Dimension is read from config to stay consistent with the embedding model.
    """
    with get_session() as session:
        session.run(_INDEX_DEVICE_VECTOR)
    logger.info(
        "Vector index on Device.embedding ensured (dimension=%d, similarity=cosine)",
        EMBEDDING_DIMENSION,
    )


def create_product_code_index() -> None:
    """
    Create a standard index on Device.product_code for filtered queries.
    """
    with get_session() as session:
        session.run(_INDEX_DEVICE_PRODUCT_CODE)
    logger.info("Index on Device.product_code ensured")


def create_decision_date_index() -> None:
    """
    Create a standard index on Device.decision_date for date-range queries.
    """
    with get_session() as session:
        session.run(_INDEX_DEVICE_DECISION_DATE)
    logger.info("Index on Device.decision_date ensured")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def initialise_schema() -> None:
    """
    Create all schema elements required by MedPredict.

    Safe to call on an already-initialised database — all statements
    use IF NOT EXISTS.
    """
    logger.info("Initialising Neo4j schema")
    create_uniqueness_constraint()
    create_vector_index()
    create_product_code_index()
    create_decision_date_index()
    logger.info("Schema initialisation complete")
