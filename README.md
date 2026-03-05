# MedPredict

510(k) Predicate Intelligence for Neurostimulation Devices

MedPredict is a knowledge graph-augmented retrieval system that helps
regulatory affairs teams identify candidate predicate devices for FDA
510(k) premarket notification submissions. It combines semantic search
over intended use statements with graph traversal of the predicate
network, and generates structured substantial equivalence analyses
grounded in real cleared submission data.

The system is scoped to neurostimulation devices and uses publicly
available FDA 510(k) data as its knowledge base.

---

## Background

A 510(k) submission is the most common pathway for bringing a new
medical device to market in the United States. The applicant must
demonstrate substantial equivalence to a previously cleared device,
known as the predicate. Selecting the right predicate shapes the
entire submission strategy and determines what performance testing
the FDA will expect.

Finding a good predicate today is largely a manual process. Regulatory
affairs specialists search the FDA's 510(k) database using keyword
search, read PDF summaries one by one, and construct equivalence
arguments by hand. This process takes days to weeks and is sensitive
to the terminology used by the searcher.

MedPredict addresses this by treating predicate search as a graph
problem. Cleared devices form a network connected by predicate
relationships. A new device should be evaluated not just against
semantically similar devices, but against the full lineage of what
those devices were themselves predicated on and what has cited them
since clearance.

---

## Architecture

The system has four layers.

The pipeline layer downloads FDA bulk data, filters to neurostimulation
product codes, extracts intended use statements from 510(k) summary
PDFs, generates sentence embeddings, and loads everything into a Neo4j
graph database.

The graph layer manages the Neo4j connection, schema, and all Cypher
queries. Device nodes store structured attributes and embedding vectors.
PREDICATED_ON edges encode the predicate network extracted from the
PREDICATENUMBER field in the FDA data.

The retrieval layer combines two mechanisms. Semantic search embeds
the user's query and finds the most similar device nodes using Neo4j's
vector index. Graph traversal then expands those seed nodes by walking
the predicate network in both directions, returning a subgraph of
ancestors and descendants.

The generation layer formats the retrieved subgraph into a structured
prompt and calls the Anthropic API to produce a ranked substantial
equivalence analysis grounded in the retrieved device data.

The Streamlit application presents a three-column interface: a query
form on the left, an interactive predicate network graph in the centre,
and the generated analysis on the right. K-numbers in the analysis are
linked directly to their FDA public records.

---

## Data Sources

All data is publicly available from the FDA.

The 510(k) bulk flat file contains structured records for every cleared
submission, including K-numbers, applicant names, device names, product
codes, decision dates, and predicate K-numbers. It is available as a
downloadable archive from the FDA's premarket FTP area.

The product classification file maps product codes to device categories
and regulatory classes.

510(k) summary PDFs contain the free text of each submission's
substantial equivalence argument, including the intended use statement.
These are hosted on the FDA website and fetched individually by
K-number.

The pipeline downloads and processes these sources automatically.

---

## Neurostimulation Scope

The system filters to the following FDA product codes by default.
These can be extended in config.py.

    GZP    Implantable spinal cord stimulator
    LLD    Deep brain stimulator
    NPN    Neurostimulator, implantable
    QFN    Implantable pulse generator for pain
    MRX    Transcutaneous electrical nerve stimulator
    IYO    Vagus nerve stimulator
    OZO    Sacral nerve stimulator
    PZI    Peripheral nerve stimulator

Only cleared submissions (SESE or SE decision codes) from 2005 onwards
are included. This keeps the dataset manageable and avoids the older
scanned PDFs that do not yield reliable text extraction.

---

## Project Structure

    med-predict/
        config.py                   Central configuration and logging
        requirements.txt
        docker-compose.yml
        .env.example

        pipeline/
            run_pipeline.py         Orchestrates all pipeline steps
            download_data.py        Downloads FDA flat files and PDFs
            filter_devices.py       Filters to neurostimulation records
            extract_text.py         PDF text extraction via pdfplumber
            parse_intended_use.py   Parses intended use statements
            extract_predicates.py   Extracts predicate edges from FDA data
            embed.py                Generates and caches embeddings
            load_graph.py           Loads nodes and edges into Neo4j

        graph/
            connection.py           Neo4j driver singleton and session context
            queries.py              All Cypher queries as named functions

        retrieval/
            semantic_search.py      Vector similarity search
            graph_traversal.py      Predicate network expansion
            retriever.py            Orchestrates retrieval pipeline

        generation/
            prompts.py              Versioned prompt templates
            generator.py            LLM API calls and response formatting

        app/
            streamlit_app.py        Two-panel Streamlit interface

        data/
            raw/                    Downloaded FDA files and PDFs
            processed/              Filtered records and extracted text
            embeddings/             Cached embedding vectors

        logs/
            medpredict.log          Rotating log file

---

## Setup

Prerequisites: Python 3.11 or higher. For the graph database, use
either a Neo4j AuraDB free tier account or run Neo4j locally with
Docker.

Create a virtual environment and install dependencies.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Copy the environment template and fill in your credentials.

    cp .env.example .env

The .env file requires the following values.

    NEO4J_URI         Connection URI from your AuraDB instance dashboard
    NEO4J_USERNAME    Usually "neo4j"
    NEO4J_PASSWORD    Set when creating the AuraDB instance
    ANTHROPIC_API_KEY Your Anthropic API key
    LLM_MODEL         Defaults to claude-sonnet-4-20250514

### Running Neo4j locally with Docker

As an alternative to AuraDB, the included docker-compose.yml starts a
local Neo4j 5 instance with APOC plugins enabled.

    docker compose up -d

The database will be available on bolt://localhost:7687 and the browser
UI at http://localhost:7474. Use these values in your .env.

    NEO4J_URI         bolt://localhost:7687
    NEO4J_USERNAME    neo4j
    NEO4J_PASSWORD    medpredict-local

---

## Running the Pipeline

The pipeline downloads data, processes it, and loads the graph. Run it
once before starting the application. Each step is idempotent and safe
to re-run if interrupted.

    python -m pipeline.run_pipeline

Pipeline steps in order:

    1. Download the FDA 510(k) flat file and product classification file
    2. Filter records to neurostimulation product codes
    3. Download 510(k) summary PDFs for filtered devices
    4. Extract text from PDFs using pdfplumber
    5. Parse intended use statements from extracted text
    6. Generate sentence embeddings using BAAI/bge-base-en-v1.5
    7. Load device nodes, predicate edges, and embeddings into Neo4j

PDF download takes the longest due to rate limiting between requests.
Expect several hours for a full neurostimulation corpus. The download
is resumable — already-downloaded files are skipped on re-run.

Embedding generation runs locally on CPU. On an M4 Mac with 24GB RAM
this takes a few minutes for a corpus of a few thousand documents.

---

## Running the Application

    streamlit run app/streamlit_app.py

The application opens in your browser at localhost:8501.

Enter a natural language description of the device you are seeking a
predicate for. Adjust the semantic candidates slider to control how
many initial matches semantic search returns, and the graph depth
slider to control how many hops the predicate traversal walks.

The graph panel shows the predicate network for the retrieved devices.
Blue nodes are semantic matches to your query. Green nodes are ancestor
devices that the seeds were predicated on. Amber nodes are descendant
devices that have cited the seeds as predicates.

The analysis panel shows the ranked substantial equivalence assessment
generated by the LLM. K-numbers in the analysis are linked to their
FDA public records.

---

## Configuration

All configurable values are in config.py. The most commonly adjusted
settings are listed below.

    NEUROSTIMULATION_PRODUCT_CODES    List of FDA product codes to include
    MIN_SUBMISSION_YEAR               Earliest submission year to include
    SEMANTIC_TOP_K                    Default number of semantic candidates
    GRAPH_TRAVERSAL_DEPTH             Default traversal depth
    EMBEDDING_MODEL_NAME              Sentence transformer model
    NEO4J_BATCH_SIZE                  Records per Neo4j write transaction
    PDF_DOWNLOAD_DELAY                Seconds between PDF requests

---

## Logging

All modules log to both the console and a rotating file at
logs/medpredict.log. Log level is set via the LOG_LEVEL environment
variable and defaults to INFO. Set it to DEBUG for detailed query and
traversal output.

There are no print statements in the codebase. All output goes through
the logging system.

---

## Embedding Model

The system uses BAAI/bge-base-en-v1.5 from the sentence-transformers
library. This model produces 768-dimensional vectors and is optimised
for semantic similarity tasks. It runs locally with no API dependency
and no cost per query.

The same model must be used at both ingestion time (pipeline) and
query time (retrieval). This is enforced by reading the model name
from config.py in both layers.

---

## Limitations

PDF coverage is approximately 60 to 70 percent of the filtered corpus.
Older submissions and scanned PDFs are skipped. Devices without an
extracted intended use statement are excluded from semantic search but
may still appear in the graph as structural nodes.

The intended use parser uses header-based section detection and may
miss statements in non-standard document layouts. Validity checks
filter out obvious failures but some noise may remain in the extracted
text.

The substantial equivalence analysis is generated by a language model
and is intended as a research aid, not a regulatory determination.
All outputs should be reviewed by a qualified regulatory affairs
professional before use in a submission.

---

## Extending to Other Device Categories

To extend the system to other FDA device categories, add the relevant
product codes to NEUROSTIMULATION_PRODUCT_CODES in config.py and
re-run the pipeline. The graph schema, retrieval logic, and generation
layer require no changes.

---

## Technology Stack

    Neo4j AuraDB          Graph database with native vector search
    sentence-transformers  Local embedding model (BAAI/bge-base-en-v1.5)
    pdfplumber             PDF text extraction
    Anthropic API          LLM generation (Claude)
    Streamlit              Web application framework
    streamlit-agraph       Interactive graph visualisation
    pandas                 Data wrangling
    python-dotenv          Environment variable management
