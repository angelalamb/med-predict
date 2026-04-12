"""
app/styles.py

CSS styles injected into the Streamlit app via st.markdown.
"""

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
