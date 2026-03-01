"""
graph/connection.py

Manages the Neo4j driver lifecycle.

Provides a single shared driver instance via get_driver(), and a
context manager for sessions.  All other graph modules import from
here rather than instantiating their own drivers.
"""

from contextlib import contextmanager

from neo4j import GraphDatabase, Session
from neo4j.exceptions import AuthError, ServiceUnavailable

from config import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    get_logger,
)

logger = get_logger(__name__)

# Module-level driver singleton — created once, reused across calls.
_driver = None


def get_driver():
    """
    Return the shared Neo4j driver, creating it if necessary.

    Uses a module-level singleton so the connection is established once
    per process rather than on every query.

    Returns:
        neo4j.Driver instance.

    Raises:
        RuntimeError: If the driver cannot connect to the database.
    """
    global _driver

    if _driver is not None:
        return _driver

    logger.info("Creating Neo4j driver for %s", NEO4J_URI)

    try:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        _driver.verify_connectivity()
        logger.info("Neo4j connection established successfully")
        return _driver

    except AuthError as exc:
        logger.error("Neo4j authentication failed: %s", exc)
        raise RuntimeError("Neo4j authentication failed — check credentials") from exc

    except ServiceUnavailable as exc:
        logger.error("Neo4j service unavailable at %s: %s", NEO4J_URI, exc)
        raise RuntimeError(f"Neo4j service unavailable at {NEO4J_URI}") from exc

    except Exception as exc:
        logger.error("Unexpected error connecting to Neo4j: %s", exc)
        raise RuntimeError(f"Neo4j connection failed: {exc}") from exc


def close_driver() -> None:
    """
    Close the shared driver and reset the singleton.

    Call this during application shutdown or at the end of a pipeline run.
    """
    global _driver

    if _driver is None:
        logger.debug("No active Neo4j driver to close")
        return

    _driver.close()
    _driver = None
    logger.info("Neo4j driver closed")


@contextmanager
def get_session() -> Session:
    """
    Context manager that yields a Neo4j session from the shared driver.

    Ensures the session is always closed after use, even if an exception
    is raised inside the with block.

    Yields:
        neo4j.Session

    Example:
        with get_session() as session:
            result = session.run("MATCH (d:Device) RETURN count(d)")
    """
    driver = get_driver()
    session = driver.session()
    try:
        yield session
    except Exception as exc:
        logger.error("Error during Neo4j session: %s", exc)
        raise
    finally:
        session.close()
        logger.debug("Neo4j session closed")
