"""
app/streamlit_app.py

MedPredict — 510(k) Predicate Intelligence for Neurostimulation Devices.

Two-panel interface:
  Left  — interactive predicate network graph visualisation
  Right — LLM-generated substantial equivalence analysis

Run with:
  streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

from config import get_logger
from generation.generator import generate
from retrieval.retriever import retrieve

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedPredict",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
    --bg:           #0d0f14;
    --surface:      #13161e;
    --surface-2:    #1a1e28;
    --border:       #252836;
    --accent:       #4f8ef7;
    --accent-dim:   #2d5ab5;
    --amber:        #f5a623;
    --green:        #3ecf8e;
    --red:          #e5534b;
    --text-primary: #e8eaf0;
    --text-muted:   #7a7f94;
    --text-dim:     #4a4f63;
    --seed-color:   #4f8ef7;
    --anc-color:    #3ecf8e;
    --desc-color:   #f5a623;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* Main layout */
.block-container {
    padding: 2rem 2.5rem 2rem 2.5rem !important;
    max-width: 100% !important;
}

/* Wordmark */
.wordmark {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.85rem;
    font-weight: 400;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin: 0;
    line-height: 1;
}
.wordmark span {
    color: var(--accent);
}
.tagline {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* Divider */
.rule {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.2rem 0 1.6rem 0;
}

/* Query input */
.stTextArea textarea {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 300 !important;
    line-height: 1.6 !important;
    padding: 0.9rem 1rem !important;
    resize: none !important;
    transition: border-color 0.15s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(79, 142, 247, 0.12) !important;
}
.stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* Button */
.stButton > button {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 7px !important;
    color: #fff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.6rem !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Panel labels */
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.75rem;
}

/* Legend */
.legend {
    display: flex;
    gap: 1.4rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
}
.legend-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Analysis panel */
.analysis-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.6rem 1.8rem;
    height: 580px;
    overflow-y: auto;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    line-height: 1.75;
    color: var(--text-primary);
}
.analysis-container h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: var(--text-primary);
    margin-top: 1.4rem;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}
.analysis-container strong {
    color: var(--accent);
    font-weight: 500;
}
.analysis-container ul, .analysis-container ol {
    padding-left: 1.4rem;
    color: var(--text-primary);
}
.analysis-container li { margin-bottom: 0.3rem; }
.analysis-container code {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    background: var(--surface-2);
    padding: 0.1em 0.4em;
    border-radius: 3px;
    color: var(--amber);
}

/* Stat chips */
.stats-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}
.stat-chip {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 0.3rem 0.75rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.04em;
}
.stat-chip span {
    color: var(--text-primary);
    font-weight: 500;
}

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 480px;
    color: var(--text-dim);
    text-align: center;
    gap: 0.6rem;
}
.empty-state-icon {
    font-size: 2.4rem;
    opacity: 0.4;
}
.empty-state-text {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: var(--text-dim);
    max-width: 260px;
    line-height: 1.6;
}

/* Error state */
.error-box {
    background: rgba(229, 83, 75, 0.08);
    border: 1px solid rgba(229, 83, 75, 0.3);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #e5534b;
    font-family: 'DM Sans', sans-serif;
}

/* Metadata footer */
.meta-footer {
    margin-top: 1rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-dim);
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}

/* Scrollbar */
.analysis-container::-webkit-scrollbar { width: 5px; }
.analysis-container::-webkit-scrollbar-track { background: transparent; }
.analysis-container::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}

/* Slider label override */
.stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
</style>
"""

# ---------------------------------------------------------------------------
# Graph visualisation helpers
# ---------------------------------------------------------------------------

# Node colours keyed by direction/role
_NODE_COLORS = {
    "seed":       "#4f8ef7",
    "ancestor":   "#3ecf8e",
    "descendant": "#f5a623",
}

_NODE_SIZE_SEED = 22
_NODE_SIZE_OTHER = 14


def _build_agraph_nodes(nodes: list[dict]) -> list[Node]:
    """
    Convert subgraph device dicts into agraph Node objects.

    Args:
        nodes: List of device dicts from the retrieval layer.

    Returns:
        List of agraph Node objects.
    """
    agraph_nodes = []
    for device in nodes:
        k = device.get("k_number", "")
        name = device.get("device_name", k)
        direction = device.get("direction", "seed")
        is_seed = device.get("is_seed", False)

        # Truncate long names for readability in the graph
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
    """
    Convert subgraph edge dicts into agraph Edge objects.

    Args:
        edges: List of {from_k, to_k} dicts from the retrieval layer.

    Returns:
        List of agraph Edge objects.
    """
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
    """
    Build the agraph physics and display configuration.

    Adjusts repulsion based on node count to avoid overcrowding.

    Args:
        node_count: Number of nodes in the graph.

    Returns:
        agraph Config object.
    """
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


# ---------------------------------------------------------------------------
# UI component renderers
# ---------------------------------------------------------------------------


def _render_header() -> None:
    """Render the MedPredict wordmark and tagline."""
    st.markdown(
        """
        <p class="wordmark">Med<span>Predict</span></p>
        <p class="tagline">510(k) Predicate Intelligence · Neurostimulation</p>
        <hr class="rule">
        """,
        unsafe_allow_html=True,
    )


def _render_query_form() -> tuple[str, int, int, bool]:
    """
    Render the query input form and return user inputs.

    Returns:
        Tuple of (query_text, top_k, depth, submitted).
    """
    query = st.text_area(
        "Device Description",
        placeholder=(
            "e.g. Implantable pulse generator for the treatment of chronic "
            "intractable pain of the trunk and limbs via spinal cord stimulation…"
        ),
        height=110,
        key="query_input",
        label_visibility="visible",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        top_k = st.slider(
            "Semantic Candidates",
            min_value=1,
            max_value=10,
            value=5,
            key="top_k",
        )
    with col_b:
        depth = st.slider(
            "Graph Depth",
            min_value=1,
            max_value=3,
            value=2,
            key="depth",
        )

    submitted = st.button("Analyse", use_container_width=True)
    return query, top_k, depth, submitted


def _render_legend() -> None:
    """Render the graph node colour legend."""
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


def _render_graph_empty() -> None:
    """Render the empty state for the graph panel."""
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


def _render_graph(subgraph: dict) -> None:
    """
    Render the predicate network graph for a retrieved subgraph.

    Args:
        subgraph: Dict with 'nodes' and 'edges' keys.
    """
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes:
        _render_graph_empty()
        return

    agraph_nodes = _build_agraph_nodes(nodes)
    agraph_edges = _build_agraph_edges(edges)
    config = _build_agraph_config(len(agraph_nodes))

    _render_legend()

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


def _render_analysis_empty() -> None:
    """Render the empty state for the analysis panel."""
    st.markdown(
        """
        <div class="analysis-container">
            <div class="empty-state">
                <div class="empty-state-icon">⚕</div>
                <div class="empty-state-text">
                    Substantial equivalence analysis will appear here
                    after you submit a device description
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_analysis_error(message: str) -> None:
    """
    Render an error message in the analysis panel.

    Args:
        message: Human-readable error description.
    """
    st.markdown(
        f'<div class="error-box">⚠ {message}</div>',
        unsafe_allow_html=True,
    )


def _build_fda_url(k_number: str) -> str:
    """
    Construct the FDA public record URL for a K-number.

    Args:
        k_number: Uppercase K-number string.

    Returns:
        FDA URL string.
    """
    return (
        f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/"
        f"pmn.cfm?ID={k_number}"
    )


def _inject_k_number_links(text: str, nodes: list[dict]) -> str:
    """
    Replace bare K-numbers in the analysis text with markdown links
    pointing to their FDA public records.

    Args:
        text: Raw analysis text from the LLM.
        nodes: Subgraph nodes used to identify valid K-numbers.

    Returns:
        Text with K-numbers replaced by markdown hyperlinks.
    """
    import re

    k_numbers = {n["k_number"] for n in nodes if "k_number" in n}

    def _replace(match: re.Match) -> str:
        k = match.group(0)
        if k in k_numbers:
            return f"[{k}]({_build_fda_url(k)})"
        return k

    return re.sub(r"\bK\d{6}\b", _replace, text)


def _render_analysis(result: dict) -> None:
    """
    Render the LLM-generated analysis with metadata footer.

    Args:
        result: Dict returned by generator.generate().
    """
    analysis = result.get("analysis", "")
    metadata = result.get("metadata", {})
    nodes = result.get("subgraph", {}).get("nodes", [])

    linked_analysis = _inject_k_number_links(analysis, nodes)

    st.markdown(
        f'<div class="analysis-container">{_markdown_to_html(linked_analysis)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="meta-footer">
            <span>model: {metadata.get('model', '—')}</span>
            <span>in: {metadata.get('input_tokens', '—')} tok</span>
            <span>out: {metadata.get('output_tokens', '—')} tok</span>
            <span>prompt: {metadata.get('prompt_version', '—')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _markdown_to_html(text: str) -> str:
    """
    Convert a markdown string to HTML for rendering inside a custom div.

    Uses the markdown library if available, otherwise returns the raw text
    wrapped in a pre tag as a safe fallback.

    Args:
        text: Markdown-formatted string.

    Returns:
        HTML string.
    """
    try:
        import markdown
        return markdown.markdown(
            text,
            extensions=["extra", "sane_lists"],
        )
    except ImportError:
        logger.warning(
            "markdown library not installed — falling back to plain text rendering"
        )
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre style='white-space:pre-wrap;font-size:0.85rem'>{escaped}</pre>"


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    """Initialise Streamlit session state keys on first load."""
    defaults = {
        "result": None,
        "error": None,
        "loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_analysis(query: str, top_k: int, depth: int) -> None:
    """
    Execute the retrieval and generation pipeline and store results
    in session state.

    Args:
        query: User's device description.
        top_k: Number of semantic candidates to retrieve.
        depth: Graph traversal depth.
    """
    st.session_state.error = None
    st.session_state.result = None

    logger.info(
        "Analysis requested | top_k=%d | depth=%d | query=%r",
        top_k,
        depth,
        query[:120],
    )

    try:
        with st.spinner("Retrieving predicate network…"):
            subgraph = retrieve(query, top_k=top_k, depth=depth)

        if not subgraph.get("nodes"):
            st.session_state.error = (
                "No matching devices found. Try broadening your device "
                "description or reducing specificity."
            )
            logger.warning("Retrieval returned empty subgraph")
            return

        with st.spinner("Generating equivalence analysis…"):
            result = generate(query, subgraph)

        st.session_state.result = result
        logger.info("Analysis complete")

    except ValueError as exc:
        st.session_state.error = str(exc)
        logger.warning("ValueError during analysis: %s", exc)

    except RuntimeError as exc:
        st.session_state.error = (
            f"A system error occurred: {exc}. "
            "Check logs for details."
        )
        logger.error("RuntimeError during analysis: %s", exc)

    except Exception as exc:
        st.session_state.error = (
            "An unexpected error occurred. Check logs for details."
        )
        logger.error("Unexpected error during analysis: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Entry point for the Streamlit application.

    Renders the full layout and handles user interactions.
    """
    st.markdown(STYLES, unsafe_allow_html=True)
    _init_session_state()

    # Header
    _render_header()

    # Input column (left, narrow) and panel columns (right, wide)
    input_col, graph_col, analysis_col = st.columns([1.1, 2.2, 2.2])

    with input_col:
        query, top_k, depth, submitted = _render_query_form()

        if submitted:
            if not query or not query.strip():
                st.markdown(
                    '<div class="error-box">Please enter a device description.</div>',
                    unsafe_allow_html=True,
                )
                logger.warning("Empty query submitted — not running analysis")
            else:
                _run_analysis(query.strip(), top_k, depth)

    with graph_col:
        st.markdown('<p class="panel-label">Predicate Network</p>', unsafe_allow_html=True)

        if st.session_state.error:
            _render_analysis_error(st.session_state.error)
        elif st.session_state.result:
            _render_graph(st.session_state.result["subgraph"])
        else:
            _render_graph_empty()

    with analysis_col:
        st.markdown('<p class="panel-label">Equivalence Analysis</p>', unsafe_allow_html=True)

        if st.session_state.error:
            _render_analysis_empty()
        elif st.session_state.result:
            _render_analysis(st.session_state.result)
        else:
            _render_analysis_empty()


if __name__ == "__main__":
    main()
