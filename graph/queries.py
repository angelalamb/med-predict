"""
graph/queries.py

All Cypher queries expressed as named Python functions.

No query string is constructed outside this module — callers pass
parameters and receive plain Python dicts or lists.  This keeps the
graph interface clean and makes queries easy to test in isolation.
"""

from config import GRAPH_TRAVERSAL_DEPTH, SEMANTIC_TOP_K, get_logger
from graph.connection import get_session

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Single-node lookups
# ---------------------------------------------------------------------------


def get_device_by_k_number(k_number: str) -> dict | None:
    """
    Fetch a single Device node by its K-number.

    Args:
        k_number: Uppercase K-number string, e.g. 'K213456'.

    Returns:
        Dict of node properties, or None if not found.
    """
    cypher = """
    MATCH (d:Device {k_number: $k_number})
    RETURN d
    """
    with get_session() as session:
        result = session.run(cypher, k_number=k_number.upper())
        record = result.single()

    if record is None:
        logger.debug("No device found for K-number %s", k_number)
        return None

    node = dict(record["d"])
    logger.debug("Retrieved device %s: %s", k_number, node.get("device_name"))
    return node


def get_devices_by_product_code(product_code: str) -> list[dict]:
    """
    Fetch all Device nodes matching a given product code.

    Args:
        product_code: Three-letter FDA product code, e.g. 'GZP'.

    Returns:
        List of device property dicts.
    """
    cypher = """
    MATCH (d:Device {product_code: $product_code})
    RETURN d
    ORDER BY d.decision_date DESC
    """
    with get_session() as session:
        result = session.run(cypher, product_code=product_code.upper())
        records = [dict(r["d"]) for r in result]

    logger.debug(
        "Found %d devices for product code %s", len(records), product_code
    )
    return records


# ---------------------------------------------------------------------------
# Predicate graph traversal
# ---------------------------------------------------------------------------


def get_ancestors(k_number: str, depth: int = GRAPH_TRAVERSAL_DEPTH) -> list[dict]:
    cypher = f"""
    MATCH path = (start:Device {{k_number: $k_number}})
                 -[:PREDICATED_ON*1..{depth}]->(ancestor:Device)
    RETURN DISTINCT ancestor,
           length(path) AS hop
    ORDER BY hop ASC
    """
    with get_session() as session:
        result = session.run(cypher, k_number=k_number.upper())
        records = [
            {{**dict(r["ancestor"]), "hop": r["hop"]}}
            for r in result
        ]
    logger.debug(
        "Found %d ancestors for %s (depth=%d)", len(records), k_number, depth
    )
    return records


def get_ancestors(k_number: str, depth: int = GRAPH_TRAVERSAL_DEPTH) -> list[dict]:
    cypher = f"""
    MATCH path = (start:Device {{k_number: $k_number}})
                 -[:PREDICATED_ON*1..{depth}]->(ancestor:Device)
    RETURN DISTINCT ancestor,
           length(path) AS hop
    ORDER BY hop ASC
    """
    with get_session() as session:
        result = session.run(cypher, k_number=k_number.upper())
        records = [
            {**dict(r["ancestor"]), "hop": r["hop"]}
            for r in result
        ]
    logger.debug(
        "Found %d ancestors for %s (depth=%d)", len(records), k_number, depth
    )
    return records


def get_descendants(k_number: str, depth: int = GRAPH_TRAVERSAL_DEPTH) -> list[dict]:
    cypher = f"""
    MATCH path = (descendant:Device)
                 -[:PREDICATED_ON*1..{depth}]->(start:Device {{k_number: $k_number}})
    RETURN DISTINCT descendant,
           length(path) AS hop
    ORDER BY hop ASC
    """
    with get_session() as session:
        result = session.run(cypher, k_number=k_number.upper())
        records = [
            {**dict(r["descendant"]), "hop": r["hop"]}
            for r in result
        ]
    logger.debug(
        "Found %d descendants for %s (depth=%d)", len(records), k_number, depth
    )
    return records


def get_predicate_lineage(
    k_number: str, depth: int = GRAPH_TRAVERSAL_DEPTH
) -> list[dict]:
    """
    Return the full predicate lineage of a device: both ancestors and
    descendants combined, deduplicated.

    Args:
        k_number: Starting device K-number.
        depth: Maximum hops in each direction.

    Returns:
        List of dicts, each with a 'direction' key ('ancestor' or 'descendant')
        and a 'hop' key indicating distance from the starting node.
    """
    ancestors = [
        {**node, "direction": "ancestor"} for node in get_ancestors(k_number, depth)
    ]
    descendants = [
        {**node, "direction": "descendant"}
        for node in get_descendants(k_number, depth)
    ]

    seen = set()
    combined = []
    for node in ancestors + descendants:
        if node["k_number"] not in seen:
            seen.add(node["k_number"])
            combined.append(node)

    logger.debug(
        "Full lineage for %s: %d unique nodes (%d ancestors, %d descendants)",
        k_number,
        len(combined),
        len(ancestors),
        len(descendants),
    )
    return combined


# ---------------------------------------------------------------------------
# Subgraph retrieval
# ---------------------------------------------------------------------------


def get_subgraph_edges(k_numbers: list[str]) -> list[dict]:
    """
    Return all PREDICATED_ON edges between a set of K-numbers.

    Used to reconstruct the subgraph structure for visualisation and
    for passing relational context to the generation layer.

    Args:
        k_numbers: List of K-numbers defining the subgraph nodes.

    Returns:
        List of dicts with keys: from_k, to_k.
    """
    cypher = """
    MATCH (a:Device)-[:PREDICATED_ON]->(b:Device)
    WHERE a.k_number IN $k_numbers
      AND b.k_number IN $k_numbers
    RETURN a.k_number AS from_k,
           b.k_number AS to_k
    """
    with get_session() as session:
        result = session.run(cypher, k_numbers=k_numbers)
        edges = [{"from_k": r["from_k"], "to_k": r["to_k"]} for r in result]

    logger.debug(
        "Found %d edges within subgraph of %d nodes",
        len(edges),
        len(k_numbers),
    )
    return edges


# ---------------------------------------------------------------------------
# Vector similarity search
# ---------------------------------------------------------------------------


def vector_similarity_search(
    embedding: list[float],
    top_k: int = SEMANTIC_TOP_K,
) -> list[dict]:
    """
    Find Device nodes whose embedding is most similar to the query embedding
    using Neo4j's built-in vector index.

    Only returns devices that have an intended_use property (i.e. those
    for which we successfully extracted and embedded text).

    Args:
        embedding: Query embedding as a list of floats.
        top_k: Number of candidates to return.

    Returns:
        List of dicts with device properties plus a 'score' key
        (cosine similarity, higher is more similar).
    """
    cypher = """
    CALL db.index.vector.queryNodes(
        'device_embedding_index',
        $top_k,
        $embedding
    )
    YIELD node AS d, score
    WHERE d.intended_use IS NOT NULL AND d.intended_use <> ''
    RETURN d, score
    ORDER BY score DESC
    """
    with get_session() as session:
        result = session.run(cypher, top_k=top_k, embedding=embedding)
        records = [
            {**dict(r["d"]), "score": r["score"]}
            for r in result
        ]

    logger.debug(
        "Vector search returned %d candidates (top score=%.4f)",
        len(records),
        records[0]["score"] if records else 0.0,
    )
    return records


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------


def count_nodes() -> int:
    """
    Return the total number of Device nodes in the graph.

    Returns:
        Integer count.
    """
    cypher = "MATCH (d:Device) RETURN count(d) AS n"
    with get_session() as session:
        result = session.run(cypher)
        count = result.single()["n"]
    logger.debug("Total Device nodes: %d", count)
    return count


def count_edges() -> int:
    """
    Return the total number of PREDICATED_ON relationships in the graph.

    Returns:
        Integer count.
    """
    cypher = "MATCH ()-[r:PREDICATED_ON]->() RETURN count(r) AS n"
    with get_session() as session:
        result = session.run(cypher)
        count = result.single()["n"]
    logger.debug("Total PREDICATED_ON edges: %d", count)
    return count


def count_nodes_with_embeddings() -> int:
    """
    Return the number of Device nodes that have an embedding vector.

    Returns:
        Integer count.
    """
    cypher = """
    MATCH (d:Device)
    WHERE d.embedding IS NOT NULL
    RETURN count(d) AS n
    """
    with get_session() as session:
        result = session.run(cypher)
        count = result.single()["n"]
    logger.debug("Device nodes with embeddings: %d", count)
    return count


def get_graph_summary() -> dict:
    """
    Return a summary dict of key graph statistics.

    Useful for health checks and logging after pipeline runs.

    Returns:
        Dict with keys: total_nodes, total_edges, nodes_with_embeddings.
    """
    summary = {
        "total_nodes": count_nodes(),
        "total_edges": count_edges(),
        "nodes_with_embeddings": count_nodes_with_embeddings(),
    }
    logger.info(
        "Graph summary: %d nodes, %d edges, %d with embeddings",
        summary["total_nodes"],
        summary["total_edges"],
        summary["nodes_with_embeddings"],
    )
    return summary
