"""
All Cypher queries expressed as named Python functions.

No query string is constructed outside this module — callers pass
parameters and receive plain Python dicts or lists.  This keeps the
graph interface clean and makes queries easy to test in isolation.
"""

from config import GRAPH_TRAVERSAL_DEPTH, SEMANTIC_TOP_K, get_logger
from graph.connection import get_session

logger = get_logger(__name__)

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

