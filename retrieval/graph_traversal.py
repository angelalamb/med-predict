"""
retrieval/graph_traversal.py

Expands a set of seed Device nodes (from semantic search) by traversing
the PREDICATED_ON graph in both directions.

Returns a structured subgraph object containing nodes and edges,
deduplicated and ready to pass to the generation layer or UI.
"""

from config import GRAPH_TRAVERSAL_DEPTH, get_logger
from graph.queries import (
    get_ancestors,
    get_descendants,
    get_subgraph_edges,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subgraph data structure
# ---------------------------------------------------------------------------


def _make_subgraph(nodes: list[dict], edges: list[dict]) -> dict:
    """
    Assemble the subgraph dict returned by traversal functions.

    Args:
        nodes: List of device property dicts.
        edges: List of {from_k, to_k} dicts.

    Returns:
        Dict with keys 'nodes' and 'edges'.
    """
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _extract_k_numbers(devices: list[dict]) -> list[str]:
    """
    Extract K-numbers from a list of device dicts.

    Args:
        devices: List of device property dicts.

    Returns:
        List of K-number strings.
    """
    return [d["k_number"] for d in devices if "k_number" in d]


def _deduplicate_nodes(nodes: list[dict]) -> list[dict]:
    """
    Remove duplicate Device nodes, keeping the first occurrence of each
    K-number.

    Args:
        nodes: List of device dicts, potentially with duplicates.

    Returns:
        Deduplicated list preserving original order.
    """
    seen = set()
    unique = []
    for node in nodes:
        k = node.get("k_number")
        if k and k not in seen:
            seen.add(k)
            unique.append(node)
    return unique


def _tag_seed_nodes(nodes: list[dict], seed_k_numbers: set[str]) -> list[dict]:
    """
    Add a boolean 'is_seed' flag to each node indicating whether it was
    a direct result of semantic search.

    Args:
        nodes: List of device dicts.
        seed_k_numbers: Set of K-numbers from semantic search results.

    Returns:
        List of dicts with 'is_seed' key added.
    """
    return [
        {**node, "is_seed": node.get("k_number") in seed_k_numbers}
        for node in nodes
    ]


# ---------------------------------------------------------------------------
# Traversal functions
# ---------------------------------------------------------------------------


def _expand_single_seed(
    seed_k_number: str,
    depth: int,
) -> list[dict]:
    """
    Traverse the predicate graph from one seed node in both directions.

    Args:
        seed_k_number: K-number of the seed device.
        depth: Number of hops to traverse in each direction.

    Returns:
        List of related device dicts with 'direction' and 'hop' keys.
    """
    ancestors = get_ancestors(seed_k_number, depth=depth)
    descendants = get_descendants(seed_k_number, depth=depth)

    logger.debug(
        "Seed %s expanded to %d ancestors and %d descendants",
        seed_k_number,
        len(ancestors),
        len(descendants),
    )

    tagged_ancestors = [
        {**node, "direction": "ancestor"} for node in ancestors
    ]
    tagged_descendants = [
        {**node, "direction": "descendant"} for node in descendants
    ]

    return tagged_ancestors + tagged_descendants


def _collect_all_nodes(
    seeds: list[dict],
    depth: int,
) -> list[dict]:
    """
    Expand all seed nodes and collect every reachable device node.

    Seed nodes themselves are included in the result with direction='seed'.

    Args:
        seeds: Semantic search result dicts (include 'score' key).
        depth: Traversal depth in each direction.

    Returns:
        Combined, deduplicated list of all device nodes in the subgraph.
    """
    seed_k_numbers = set(_extract_k_numbers(seeds))

    # Start with the seeds themselves
    all_nodes = [
        {**seed, "direction": "seed"}
        for seed in seeds
    ]

    for seed in seeds:
        k_number = seed.get("k_number")
        if not k_number:
            logger.warning("Seed device missing k_number — skipping expansion")
            continue

        related = _expand_single_seed(k_number, depth=depth)
        all_nodes.extend(related)

    deduped = _deduplicate_nodes(all_nodes)
    tagged = _tag_seed_nodes(deduped, seed_k_numbers)

    logger.debug(
        "Collected %d unique nodes from %d seeds after traversal",
        len(tagged),
        len(seeds),
    )
    return tagged


def _fetch_subgraph_edges(nodes: list[dict]) -> list[dict]:
    """
    Fetch all PREDICATED_ON edges that exist between nodes in the subgraph.

    Args:
        nodes: All nodes in the subgraph.

    Returns:
        List of {from_k, to_k} edge dicts.
    """
    k_numbers = _extract_k_numbers(nodes)
    if not k_numbers:
        return []

    edges = get_subgraph_edges(k_numbers)
    logger.debug(
        "Retrieved %d edges within subgraph of %d nodes",
        len(edges),
        len(k_numbers),
    )
    return edges


def _log_subgraph_summary(subgraph: dict) -> None:
    """
    Log a summary of the assembled subgraph.

    Args:
        subgraph: Dict with 'nodes' and 'edges' keys.
    """
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]
    seeds = [n for n in nodes if n.get("is_seed")]
    logger.info(
        "Subgraph assembled: %d nodes (%d seeds), %d edges",
        len(nodes),
        len(seeds),
        len(edges),
    )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def expand(
    seeds: list[dict],
    depth: int = GRAPH_TRAVERSAL_DEPTH,
) -> dict:
    """
    Expand a list of seed devices into a full subgraph by traversing the
    PREDICATED_ON graph in both directions.

    Args:
        seeds: List of device dicts from semantic search, each containing
               at minimum a 'k_number' key.
        depth: Number of hops to traverse from each seed in each direction.
               Defaults to GRAPH_TRAVERSAL_DEPTH from config.

    Returns:
        Subgraph dict with two keys:
          'nodes' — list of all device dicts in the subgraph, each with:
                    direction ('seed' | 'ancestor' | 'descendant'),
                    is_seed (bool),
                    and all standard device properties.
          'edges' — list of {from_k, to_k} dicts for edges within the
                    subgraph.
    """
    if not seeds:
        logger.warning("No seed nodes provided for graph traversal")
        return _make_subgraph(nodes=[], edges=[])

    logger.info(
        "Expanding %d seed nodes with depth=%d", len(seeds), depth
    )

    nodes = _collect_all_nodes(seeds, depth=depth)
    edges = _fetch_subgraph_edges(nodes)
    subgraph = _make_subgraph(nodes, edges)

    _log_subgraph_summary(subgraph)

    return subgraph
