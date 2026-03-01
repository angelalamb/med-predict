"""
retrieval/retriever.py

Public entry point for the retrieval layer.

Orchestrates semantic search and graph traversal into a single call
that returns a fully assembled subgraph ready for the generation layer
and the UI.

All other layers (generation, app) import from here rather than
calling semantic_search or graph_traversal directly.
"""

import time

from config import GRAPH_TRAVERSAL_DEPTH, SEMANTIC_TOP_K, get_logger
from retrieval.graph_traversal import expand
from retrieval.semantic_search import search

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result validation
# ---------------------------------------------------------------------------


def _has_intended_use(node: dict) -> bool:
    """
    Check whether a node has a non-empty intended use statement.

    Args:
        node: Device property dict.

    Returns:
        True if intended_use is present and non-empty.
    """
    intended_use = node.get("intended_use", "")
    return bool(intended_use and intended_use.strip())


def _filter_nodes_without_intended_use(subgraph: dict) -> dict:
    """
    Remove non-seed nodes that have no intended use text from the subgraph.

    Seed nodes are always kept regardless, since they were directly
    matched by the query and may still be useful as structural anchors
    in the graph visualisation.

    Args:
        subgraph: Dict with 'nodes' and 'edges' keys.

    Returns:
        Subgraph with filtered nodes (edges are unchanged — the UI and
        generation layer handle missing endpoints gracefully).
    """
    before = len(subgraph["nodes"])
    filtered = [
        node for node in subgraph["nodes"]
        if node.get("is_seed") or _has_intended_use(node)
    ]
    after = len(filtered)

    if before != after:
        logger.debug(
            "Filtered %d nodes without intended use text (%d → %d)",
            before - after,
            before,
            after,
        )

    return {**subgraph, "nodes": filtered}


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _log_retrieval_time(start: float, query: str, subgraph: dict) -> None:
    """
    Log end-to-end retrieval time and subgraph summary.

    Args:
        start: Unix timestamp from time.time() at retrieval start.
        query: Original query string.
        subgraph: Assembled subgraph dict.
    """
    elapsed = time.time() - start
    logger.info(
        "Retrieval complete in %.2fs | query=%r | nodes=%d | edges=%d",
        elapsed,
        query[:80],
        len(subgraph["nodes"]),
        len(subgraph["edges"]),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int = SEMANTIC_TOP_K,
    depth: int = GRAPH_TRAVERSAL_DEPTH,
) -> dict:
    """
    Retrieve a subgraph of related Device nodes for a natural language query.

    Pipeline:
      1. Embed the query using the same model as ingestion
      2. Run vector similarity search to find seed candidates
      3. Expand seeds by traversing the PREDICATED_ON graph
      4. Return the assembled subgraph

    Args:
        query: Natural language description of the device or intended use.
        top_k: Number of seed candidates from semantic search.
        depth: Number of hops to traverse from each seed.

    Returns:
        Dict with two keys:
          'nodes' — list of device dicts, each containing:
                      k_number, device_name, applicant, product_code,
                      decision_date, intended_use, score (seeds only),
                      direction ('seed' | 'ancestor' | 'descendant'),
                      is_seed (bool)
          'edges' — list of {from_k, to_k} dicts representing
                    PREDICATED_ON relationships within the subgraph

        Returns {'nodes': [], 'edges': []} if no results found.
    """
    start = time.time()

    logger.info(
        "Starting retrieval | top_k=%d | depth=%d | query=%r",
        top_k,
        depth,
        query[:120],
    )

    seeds = search(query, top_k=top_k)

    if not seeds:
        logger.warning("Semantic search returned no results for query: %r", query[:120])
        return {"nodes": [], "edges": []}

    subgraph = expand(seeds, depth=depth)
    subgraph = _filter_nodes_without_intended_use(subgraph)

    _log_retrieval_time(start, query, subgraph)

    return subgraph
