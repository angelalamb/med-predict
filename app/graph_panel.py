"""
app/graph_panel.py

Builds and renders the interactive predicate network graph (left panel).
"""

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from config import get_logger

logger = get_logger(__name__)

_NODE_COLORS = {
    "seed":       "#4f8ef7",
    "ancestor":   "#3ecf8e",
    "descendant": "#f5a623",
}

_NODE_SIZE_SEED = 22
_NODE_SIZE_OTHER = 14


def _build_agraph_nodes(nodes: list[dict]) -> list[Node]:
    agraph_nodes = []
    for device in nodes:
        k = device.get("k_number", "")
        name = device.get("device_name", k)
        direction = device.get("direction", "seed")
        is_seed = device.get("is_seed", False)

        label = name[:20] + "…" if len(name) > 20 else name
        color = _NODE_COLORS.get(direction, "#7a7f94")
        size = _NODE_SIZE_SEED if is_seed else _NODE_SIZE_OTHER

        tooltip = (
            f"{k}\n"
            f"{name}\n"
            f"{device.get('applicant', '')}\n"
            f"Cleared: {device.get('decision_date', '')}"
        )

        agraph_nodes.append(
            Node(
                id=k,
                label=label,
                size=size,
                color=color,
                title=tooltip,
                font={
                    "color": "#ffffff",
                    "size": 13,
                    "face": "DM Sans",
                    "background": "rgba(13, 15, 20, 0.65)",
                    "strokeWidth": 0,
                },
            )
        )

    return agraph_nodes


def _build_agraph_edges(edges: list[dict]) -> list[Edge]:
    return [
        Edge(
            source=e["from_k"],
            target=e["to_k"],
            color="#252836",
            width=1.5,
        )
        for e in edges
    ]


def _build_agraph_config(node_count: int) -> Config:
    spring_length = min(200 + node_count * 4, 400)

    return Config(
        width="100%",
        height=520,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#4f8ef7",
        collapsible=False,
        node={
            "labelProperty": "label",
            "renderLabel": True,
            "fontSize": 13,
            "fontColor": "#ffffff",
            "fontFamily": "DM Sans",
        },
        link={
            "labelProperty": "label",
            "renderLabel": False,
        },
        d3={
            "gravity": -500,
            "linkLength": spring_length,
            "linkStrength": 0.4,
            "alphaTarget": 0.05,
        },
    )


def render_legend() -> None:
    st.markdown(
        """
        <div class="legend">
            <div class="legend-item">
                <div class="legend-dot" style="background:#4f8ef7;"></div>
                Seed (semantic match)
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:#3ecf8e;"></div>
                Ancestor (upstream predicate)
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background:#f5a623;"></div>
                Descendant (downstream citation)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph_empty() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">◎</div>
            <div class="empty-state-text">
                Predicate network will appear here after analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_graph(subgraph: dict) -> None:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes:
        render_graph_empty()
        return

    agraph_nodes = _build_agraph_nodes(nodes)
    agraph_edges = _build_agraph_edges(edges)
    config = _build_agraph_config(len(agraph_nodes))

    render_legend()

    seed_count = sum(1 for n in nodes if n.get("is_seed"))
    st.markdown(
        f"""
        <div class="stats-row">
            <div class="stat-chip"><span>{len(nodes)}</span> nodes</div>
            <div class="stat-chip"><span>{len(edges)}</span> edges</div>
            <div class="stat-chip"><span>{seed_count}</span> seeds</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
    logger.debug("Graph rendered with %d nodes and %d edges", len(nodes), len(edges))
