# ---------------------------------------------------------------------------
# MedPredict — single image for both the API and the Streamlit app.
# The service is selected by overriding CMD in docker-compose / render.yaml.
#
# Build:  docker build -t medpredict .
# Run API:       docker run -p 8000:8000 --env-file .env medpredict
# Run Streamlit: docker run -p 8501:8501 --env-file .env medpredict \
#                  streamlit run app/streamlit_app.py \
#                    --server.port 8501 --server.address 0.0.0.0 \
#                    --server.headless true
# ---------------------------------------------------------------------------

FROM python:3.12-slim

# ---- System dependencies ---------------------------------------------------
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Non-root user ---------------------------------------------------------
# Running as root inside a container is a security risk.
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# ---- Python dependencies ---------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Pre-download embedding model ------------------------------------------
# Baking the model into the image avoids a slow first request in production.
# The model is ~400 MB; set a known cache location owned by appuser.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')" \
    && chown -R appuser:appuser /app/.cache

# ---- Application code ------------------------------------------------------
COPY . .
RUN chown -R appuser:appuser /app

# ---- Runtime ---------------------------------------------------------------
USER appuser

# Expose both service ports so docker-compose can map them
EXPOSE 8000
EXPOSE 8501

# Default: run the API. Override in docker-compose / render.yaml for Streamlit.
CMD ["python3", "api/main.py"]
