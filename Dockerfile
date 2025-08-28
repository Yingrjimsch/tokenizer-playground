# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG INSTALL_HF=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep HF caches ephemeral/small and reduce threads
    HF_HOME=/tmp/hf \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_MAX_THREADS=1

WORKDIR /app

# Install system dependencies only if needed; keep image slim
# Slim base already has certs in most cases; keep minimal
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt ./
COPY requirements-hf.txt ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_HF" = "true" ]; then pip install --no-cache-dir -r requirements-hf.txt; fi

# Copy application code
COPY . .

# Streamlit config (also present in repo under .streamlit/config.toml)
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
