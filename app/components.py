"""
app/components.py

Renders the header, query form, and analysis panel (right panel).
"""

import re

import streamlit as st

import config


def render_header() -> None:
    st.markdown(
        """
        <p class="wordmark">Med<span>Predict</span></p>
        <p class="tagline">510(k) Predicate Intelligence · Diagnostic Ultrasound</p>
        <hr class="rule">
        """,
        unsafe_allow_html=True,
    )


def render_query_form() -> tuple[str, int, int, list[str] | None, bool]:
    """
    Render the query input form.

    Returns:
        Tuple of (query_text, top_k, depth, categories, submitted).
        categories is None when Search All is used, or a list of category
        keys when Search Selected is used.
    """
    query = st.text_area(
        "Device Description",
        placeholder=(
            "e.g. Portable diagnostic ultrasound system for abdominal and "
            "obstetric imaging using pulsed echo technology…"
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

    st.markdown("**Filters**")
    checked = {
        key: st.checkbox(cat["label"], value=True, key=f"cat_{key}")
        for key, cat in config.DEVICE_CATEGORIES.items()
    }

    col_c, col_d = st.columns(2)
    with col_c:
        search_all = st.button("Search All", use_container_width=True)
    with col_d:
        search_selected = st.button("Search Selected", use_container_width=True)

    if search_all:
        return query, top_k, depth, None, True
    if search_selected:
        selected = [key for key, is_checked in checked.items() if is_checked]
        return query, top_k, depth, selected or None, True
    return query, top_k, depth, None, False


def render_analysis_empty() -> None:
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


def render_analysis_error(message: str) -> None:
    st.markdown(
        f'<div class="error-box">⚠ {message}</div>',
        unsafe_allow_html=True,
    )


def render_analysis(result: dict) -> None:
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


def _build_fda_url(k_number: str) -> str:
    return (
        f"https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/"
        f"pmn.cfm?ID={k_number}"
    )


def _inject_k_number_links(text: str, nodes: list[dict]) -> str:
    k_numbers = {n["k_number"] for n in nodes if "k_number" in n}

    def _replace(match: re.Match) -> str:
        k = match.group(0)
        if k in k_numbers:
            return f"[{k}]({_build_fda_url(k)})"
        return k

    return re.sub(r"\bK\d{6}\b", _replace, text)


def _markdown_to_html(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(text, extensions=["extra", "sane_lists"])
    except ImportError:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre style='white-space:pre-wrap;font-size:0.85rem'>{escaped}</pre>"
