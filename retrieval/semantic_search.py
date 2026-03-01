"""
retrieval/semantic_search.py

Embeds a natural language query using the same model used at ingestion
time, then runs a vector similarity search against the Neo4j device
embedding index.

Returns the top-K Device nodes most semantically similar to the query.
"""

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, SEMANTIC_TOP_K, get_logger
from graph.queries import vector_similarity_search

logger = get_logger(__name__)

# Module-level model singleton — loaded once, reused across queries.
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Return the shared embedding model, loading it on first call.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        RuntimeError: If the model cannot be loaded.
    """
    global _model

    if _model is not None:
        return _model

    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    try:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
        return _model
    except Exception as exc:
        logger.error("Failed to load embedding model %s: %s", EMBEDDING_MODEL_NAME, exc)
        raise RuntimeError(f"Could not load embedding model: {exc}") from exc


def _embed_query(query: str) -> list[float]:
    """
    Encode a query string into an embedding vector.

    Uses the same model and normalisation settings as the ingestion
    pipeline to ensure vectors are comparable.

    Args:
        query: Natural language device description.

    Returns:
        Embedding as a list of floats.
    """
    model = _get_model()
    embedding = model.encode(
        query,
        normalize_embeddings=True,  # Must match ingestion settings
        show_progress_bar=False,
    )
    logger.debug(
        "Encoded query (%d chars) to vector of dimension %d",
        len(query),
        len(embedding),
    )
    return embedding.tolist()


def _log_candidates(candidates: list[dict]) -> None:
    """
    Log the top candidates returned by vector search at DEBUG level.

    Args:
        candidates: List of device dicts with a 'score' key.
    """
    for idx, candidate in enumerate(candidates, start=1):
        logger.debug(
            "  Candidate %d: %s | %s | score=%.4f",
            idx,
            candidate.get("k_number"),
            candidate.get("device_name", "")[:60],
            candidate.get("score", 0.0),
        )


def search(query: str, top_k: int = SEMANTIC_TOP_K) -> list[dict]:
    """
    Embed a query and return the most semantically similar Device nodes.

    Args:
        query: Natural language description of a device or intended use.
        top_k: Number of candidate devices to return.

    Returns:
        List of device property dicts ordered by similarity score descending.
        Each dict includes a 'score' key (cosine similarity, 0–1).
    """
    if not query or not query.strip():
        logger.warning("Empty query passed to semantic search — returning no results")
        return []

    logger.info(
        "Running semantic search (top_k=%d) for query: %r",
        top_k,
        query[:120],
    )

    embedding = _embed_query(query)
    candidates = vector_similarity_search(embedding, top_k=top_k)

    logger.info(
        "Semantic search returned %d candidates",
        len(candidates),
    )
    _log_candidates(candidates)

    return candidates
