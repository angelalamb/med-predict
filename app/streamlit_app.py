"""
app/streamlit_app.py

MedPredict — 510(k) Predicate Intelligence for Diagnostic Ultrasound Devices.

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

from app.components import (
    render_analysis,
    render_analysis_empty,
    render_analysis_error,
    render_header,
    render_query_form,
)
from app.graph_panel import render_graph, render_graph_empty
from app.styles import STYLES
from config import get_logger
from generation.generator import generate
from retrieval.retriever import retrieve

logger = get_logger(__name__)

# Page configuration — must be the first Streamlit call
st.set_page_config(
    page_title="MedPredict",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    defaults = {
        "result": None,
        "error": None,
        "loading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_analysis(query: str, top_k: int, depth: int) -> None:
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
            f"A system error occurred: {exc}. Check logs for details."
        )
        logger.error("RuntimeError during analysis: %s", exc)

    except Exception as exc:
        st.session_state.error = (
            "An unexpected error occurred. Check logs for details."
        )
        logger.error("Unexpected error during analysis: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    _init_session_state()

    render_header()

    input_col, graph_col, analysis_col = st.columns([1.1, 2.2, 2.2])

    with input_col:
        query, top_k, depth, submitted = render_query_form()

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
            render_analysis_error(st.session_state.error)
        elif st.session_state.result:
            render_graph(st.session_state.result["subgraph"])
        else:
            render_graph_empty()

    with analysis_col:
        st.markdown('<p class="panel-label">Equivalence Analysis</p>', unsafe_allow_html=True)

        if st.session_state.error:
            render_analysis_empty()
        elif st.session_state.result:
            render_analysis(st.session_state.result)
        else:
            render_analysis_empty()


if __name__ == "__main__":
    main()
