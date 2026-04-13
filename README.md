# MedPredict

510(k) Predicate Intelligence for Medical Devices

MedPredict is a knowledge graph-augmented retrieval system that helps
regulatory affairs teams identify candidate predicate devices for FDA
510(k) premarket notification submissions. It combines semantic search
over intended use statements with graph traversal of the predicate
network, and generates structured substantial equivalence analyses
grounded in real cleared submission data.

The system covers three device categories: diagnostic ultrasound,
AI/ML radiology software, and wearable continuous monitors. It uses
publicly available FDA 510(k) data as its knowledge base.

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

The system has five layers.

The pipeline layer downloads FDA bulk data, filters to the configured
device categories, extracts intended use statements from 510(k)
summary PDFs, generates sentence embeddings, and loads everything into
a Neo4j graph database.

The graph layer manages the Neo4j connection, schema, and all Cypher
queries. Device nodes store structured attributes, embedding vectors,
and a category tag. PREDICATED_ON edges encode the predicate network
extracted from the PREDICATENUMBER field in the FDA data.

The retrieval layer combines two mechanisms. Semantic search embeds
the user's query and finds the most similar device nodes using Neo4j's
vector index, optionally filtered to selected categories. Graph
traversal then expands those seed nodes by walking the predicate
network in both directions, returning a subgraph of ancestors and
descendants.

The generation layer formats the retrieved subgraph into a structured
prompt and calls the Anthropic API to produce a ranked substantial
equivalence analysis grounded in the retrieved device data.

The API layer exposes the retrieval and generation pipeline as an
authenticated REST API built with FastAPI. Endpoints are rate-limited
and require an API key.

The Streamlit application presents a two-panel interface: an
interactive predicate network graph on the left and the generated
analysis on the right. K-numbers in the analysis are linked directly
to their FDA public records. Category filters allow scoping the search
to one or more device types.

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

## Device Categories

The system supports the following device categories, defined in
config.py. Each category specifies product codes and optional
include/exclude term filters for mixed-code categories.

**Ultrasound Imaging**

    IYO    System, Imaging, Pulsed Echo, Ultrasonic
    IYN    System, Imaging, Pulsed Doppler, Ultrasonic
    ITX    Transducer, Ultrasonic, Diagnostic

**AI/ML Radiology**

    QFM    Radiological Computer-Assisted Prioritization (CADt)
    QAS    Radiological Computer-Assisted Triage and Notification
    QBS    Computer-Assisted Detection/Diagnosis — Fracture
    QDQ    Computer-Assisted Detection/Diagnosis — Cancer
    MYN    Medical Image Analyzers (CADe detection)
    POK    Computer-Assisted Diagnostic Software for Cancer (CADx)

**Wearables & Continuous Monitors**

    DPS    Electrocardiograph (wearable ECG, filtered by device name)
    DQA    Electrocardiograph (filtered by device name)
    LNB    Glucose monitoring systems (filtered by device name)
    NBW    Blood glucose meter (filtered by device name)

Mixed-code categories like wearables use include/exclude term
filtering on the device name to retain only relevant devices.

Only cleared submissions (SESE or SE decision codes) from 2005 onwards
are included across all categories.

---

## Project Structure

    med-predict/
        config.py                   Central configuration and logging
        requirements.txt
        Dockerfile
        docker-compose.yml          Local dev: API + Streamlit + Neo4j
        render.yaml                 Render deployment configuration

        api/
            main.py                 FastAPI app setup and entry point
            routes.py               API endpoint handlers
            models.py               Pydantic request/response schemas
            auth.py                 API key authentication
            limiter.py              Rate limiter configuration

        app/
            streamlit_app.py        Two-panel Streamlit interface

        pipeline/
            run_pipeline.py         Orchestrates all pipeline steps
            download_data.py        Downloads FDA flat files and PDFs
            filter_devices.py       Filters records by category
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

        data/
            raw/                    Downloaded FDA files and PDFs
            processed/              Filtered records and extracted text
            embeddings/             Cached embedding vectors

        logs/
            medpredict.log          Rotating log file

---

## Setup

Prerequisites: Python 3.11 or higher. A Neo4j AuraDB free tier account
for the graph database, and an Anthropic API key for generation.

Create a virtual environment and install dependencies.

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Create a .env file in the project root with the following values.

    NEO4J_URI         Connection URI from your AuraDB instance
                      (e.g. neo4j+s://xxxxxxxx.databases.neo4j.io)
    NEO4J_USERNAME    Usually "neo4j"
    NEO4J_PASSWORD    Set when creating the AuraDB instance
    ANTHROPIC_API_KEY Your Anthropic API key
    API_KEY           Secret key required by the API (X-API-Key header)
    LLM_MODEL         Defaults to claude-sonnet-4-20250514
    LOG_LEVEL         Defaults to INFO

Generate a strong API key with:

    python3 -c "import secrets; print(secrets.token_urlsafe(32))"

---

## Local Development with Docker

The included docker-compose.yml starts Neo4j, the API, and the
Streamlit app together. Secrets are read from your .env file.

    docker compose up --build

Services:

    API            http://localhost:8000  (docs at /docs)
    Streamlit app  http://localhost:8501
    Neo4j browser  http://localhost:7474

To start only Neo4j and run the Python services natively:

    docker compose up neo4j

The Neo4j container uses bolt://localhost:7687. Set this as NEO4J_URI
in your .env when running natively.

---

## Running the Pipeline

The pipeline downloads data, processes it, and loads the graph. Run it
once before starting the application, pointing your .env at the target
Neo4j instance (local or AuraDB). Each step is idempotent and safe to
re-run if interrupted. The graph is additive — running multiple
categories populates Neo4j without overwriting existing nodes.

Run a single category:

    python -m pipeline.run_pipeline --category ultrasound
    python -m pipeline.run_pipeline --category ai_ml_radiology
    python -m pipeline.run_pipeline --category wearables

Run all categories in sequence:

    python -m pipeline.run_pipeline --all

Pipeline steps in order:

    1. Download the FDA 510(k) flat file and product classification file
    2. Filter records to the category's product codes and term filters
    3. Download 510(k) summary PDFs for filtered devices
    4. Extract text from PDFs using pdfplumber
    5. Parse intended use statements from extracted text
    6. Generate sentence embeddings using BAAI/bge-base-en-v1.5
    7. Extract predicate edges
    8. Load device nodes, predicate edges, and embeddings into Neo4j

PDF download takes the longest due to rate limiting between requests.
Embedding generation runs locally on CPU. Both steps are resumable —
already-downloaded files and cached embeddings are skipped on re-run.

---

## Running the Application

**Streamlit app**

    streamlit run app/streamlit_app.py

Opens in your browser at http://localhost:8501.

**API**

    python3 api/main.py

Runs on http://localhost:8000. Interactive docs at http://localhost:8000/docs.

All API endpoints except /health require an X-API-Key header.

    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your-api-key" \
      -d '{"query": "portable diagnostic ultrasound system for abdominal imaging", "k": 5}'

To filter results to specific categories:

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -H "X-API-Key: your-api-key" \
      -d '{"query": "wearable ECG monitor", "k": 5, "categories": ["wearables"]}'

---

## Deployment

The project deploys to Render using the included render.yaml, which
defines two web services built from the same Dockerfile:

    medpredict-api   FastAPI backend
    medpredict-app   Streamlit frontend

Both services connect to Neo4j AuraDB. Neither requires a local Neo4j
instance in production.

**Deploy steps:**

1. Push the repo to GitHub
2. Create a Render account and connect the GitHub repo
3. Render detects render.yaml and creates both services automatically
4. In the Render dashboard, set the following environment variables
   for each service before deploying:

   Both services:
       NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, ANTHROPIC_API_KEY

   API service only:
       API_KEY

   Streamlit service only:
       API_URL (public URL of the API service), API_KEY

5. Trigger a deploy

---

## API Reference

    GET  /health        Service and dependency health check (no auth)
    POST /query         Query devices by natural language description
    GET  /stats         Usage statistics placeholder

Authentication: include X-API-Key header with all requests except /health.

Rate limits: /query is limited to 20 requests per hour per IP.
/stats is limited to 10 requests per hour per IP.

Full interactive documentation is available at /docs when the API is running.

---

## Configuration

All configurable values are in config.py. The most commonly adjusted
settings are listed below.

    DEVICE_CATEGORIES             Device category definitions with product
                                  codes, labels, and term filters
    MIN_SUBMISSION_YEAR           Earliest submission year to include
    SEMANTIC_TOP_K                Default number of semantic candidates
    GRAPH_TRAVERSAL_DEPTH         Default traversal depth
    EMBEDDING_MODEL_NAME          Sentence transformer model
    NEO4J_BATCH_SIZE              Records per Neo4j write transaction
    PDF_DOWNLOAD_DELAY            Seconds between PDF requests
    CLAUDE_INPUT_TOKEN_COST       Per-token cost for input (USD)
    CLAUDE_OUTPUT_TOKEN_COST      Per-token cost for output (USD)

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

The model is loaded lazily on the first query, not at startup. In the
Docker image it is pre-downloaded during the build so the first request
does not incur a download delay.

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

To add a new device category, add an entry to DEVICE_CATEGORIES in
config.py with the relevant product codes, label, and optional term
filters. Then run the pipeline for the new category:

    python -m pipeline.run_pipeline --category your_new_category

The graph schema, retrieval logic, and generation layer require no
changes. The new category will automatically appear in the Streamlit
UI filter panel.

---

## Technology Stack

    Neo4j AuraDB           Graph database with native vector search
    sentence-transformers  Local embedding model (BAAI/bge-base-en-v1.5)
    pdfplumber             PDF text extraction
    Anthropic API          LLM generation (Claude)
    FastAPI                REST API framework
    Streamlit              Web application framework
    streamlit-agraph       Interactive graph visualisation
    pandas                 Data wrangling
    python-dotenv          Environment variable management
    Render                 Cloud deployment
    Docker                 Containerisation
