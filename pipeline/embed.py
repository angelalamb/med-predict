"""
Generates sentence embeddings for intended use statements using
sentence-transformers and stores them in a local cache.

Embeddings are later loaded by load_graph.py to populate Neo4j node
properties, and at query time by the retrieval layer.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_CACHE_PATH,
    INTENDED_USE_PATH,
    get_logger,
)

logger = get_logger(__name__)


def _load_model(model_name: str) -> SentenceTransformer:
    """
    Load the sentence-transformers model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        RuntimeError: If model loading fails.
    """
    logger.info("Loading embedding model: %s", model_name)
    try:
        model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
        return model
    except Exception as exc:
        logger.error("Failed to load model %s: %s", model_name, exc)
        raise RuntimeError(f"Could not load embedding model: {exc}") from exc


def _load_cache(cache_path: Path) -> dict[str, list[float]]:
    """
    Load the embeddings cache from disk if it exists.

    Args:
        cache_path: Path to the pickle cache file.

    Returns:
        Dict mapping K-number → embedding list, or empty dict.
    """
    if not cache_path.exists():
        logger.info("No existing embeddings cache found — starting fresh")
        return {}

    try:
        with cache_path.open("rb") as fh:
            cache = pickle.load(fh)
        logger.info(
            "Loaded embeddings cache with %d entries from %s",
            len(cache),
            cache_path,
        )
        return cache
    except (pickle.UnpicklingError, OSError) as exc:
        logger.warning(
            "Could not load embeddings cache from %s: %s — starting fresh",
            cache_path,
            exc,
        )
        return {}


def _save_cache(cache: dict[str, list[float]], cache_path: Path) -> None:
    """
    Persist the embeddings cache to disk.

    Args:
        cache: Dict mapping K-number → embedding list.
        cache_path: Destination path.
    """
    with cache_path.open("wb") as fh:
        pickle.dump(cache, fh)
    logger.info(
        "Saved embeddings cache (%d entries) to %s",
        len(cache),
        cache_path,
    )


def _load_intended_use_df() -> pd.DataFrame:
    """
    Load the intended use CSV produced by parse_intended_use.py.

    Returns:
        DataFrame with k_number and intended_use_text columns.

    Raises:
        FileNotFoundError: If the intended use CSV does not exist.
    """
    if not INTENDED_USE_PATH.exists():
        raise FileNotFoundError(
            f"Intended use file not found at {INTENDED_USE_PATH}. "
            "Run parse_intended_use.parse_intended_use() first."
        )

    df = pd.read_csv(INTENDED_USE_PATH, dtype=str)
    df = df.dropna(subset=["k_number", "intended_use_text"])
    logger.info(
        "Loaded %d intended use statements from %s",
        len(df),
        INTENDED_USE_PATH,
    )
    return df


def _identify_uncached(
    df: pd.DataFrame,
    cache: dict[str, list[float]],
) -> pd.DataFrame:
    """
    Filter the DataFrame to rows that do not yet have cached embeddings.

    Args:
        df: Full intended use DataFrame.
        cache: Existing embeddings cache.

    Returns:
        DataFrame containing only rows needing new embeddings.
    """
    needs_embedding = df[~df["k_number"].isin(cache)].copy()
    logger.info(
        "%d entries already cached; %d need embedding",
        len(df) - len(needs_embedding),
        len(needs_embedding),
    )
    return needs_embedding


def _embed_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """
    Encode a list of texts in batches and return the stacked embeddings.

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of strings to encode.
        batch_size: Number of texts to encode per batch.

    Returns:
        2D numpy array of shape (len(texts), embedding_dim).
    """
    all_embeddings = []
    total = len(texts)

    for batch_start in range(0, total, batch_size):
        batch = texts[batch_start : batch_start + batch_size]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,  # Required for BGE models
        )
        all_embeddings.append(embeddings)

        processed = min(batch_start + batch_size, total)
        logger.info(
            "Embedding progress: %d/%d texts processed",
            processed,
            total,
        )

    return np.vstack(all_embeddings)


def generate_embeddings() -> dict[str, list[float]]:
    """
    Generate embeddings for all intended use statements not yet in cache.

    Loads the intended use CSV, skips already-cached K-numbers,
    encodes new texts in batches, updates the cache, and saves it.

    Returns:
        Complete embeddings cache dict mapping K-number → embedding list.
    """
    df = _load_intended_use_df()
    cache = _load_cache(EMBEDDINGS_CACHE_PATH)

    to_embed = _identify_uncached(df, cache)

    if to_embed.empty:
        logger.info("All embeddings are already cached — nothing to do")
        return cache

    model = _load_model(EMBEDDING_MODEL_NAME)

    texts = to_embed["intended_use_text"].tolist()
    k_numbers = to_embed["k_number"].tolist()

    logger.info("Generating embeddings for %d texts", len(texts))

    embeddings_array = _embed_in_batches(model, texts, EMBEDDING_BATCH_SIZE)

    for k_number, embedding in zip(k_numbers, embeddings_array):
        cache[k_number] = embedding.tolist()

    _save_cache(cache, EMBEDDINGS_CACHE_PATH)

    logger.info(
        "Embedding generation complete. Cache now contains %d entries",
        len(cache),
    )
    return cache


def load_cached_embeddings() -> dict[str, list[float]]:
    """
    Load the embeddings cache from disk without regenerating.

    Useful for the retrieval layer which needs embeddings at query time
    without re-running the full pipeline.

    Returns:
        Dict mapping K-number → embedding list.

    Raises:
        FileNotFoundError: If no cache exists yet.
    """
    if not EMBEDDINGS_CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings cache not found at {EMBEDDINGS_CACHE_PATH}. "
            "Run generate_embeddings() first."
        )
    return _load_cache(EMBEDDINGS_CACHE_PATH)
