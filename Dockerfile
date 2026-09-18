# Multi-stage Dockerfile for LegalRAG API
# Configured for standard container runtimes and Hugging Face Docker Spaces (port 7860, non-root user)

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ENVIRONMENT=local_stub \
    API_PORT=7860 \
    API_HOST=0.0.0.0 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face Spaces default UID 1000)
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Copy dependency specifications first to leverage Docker layer caching
COPY --chown=user:user requirements.txt pyproject.toml README.md ./

# Install Python dependencies and CPU-optimized FAISS
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install faiss-cpu>=1.7.4

# Copy application source code and scripts
COPY --chown=user:user src/ ./src/
COPY --chown=user:user scripts/ ./scripts/

# Install the package itself
RUN pip install --no-deps -e .

# Create artifacts directory with appropriate permissions
RUN mkdir -p artifacts && chown -R user:user artifacts

# Switch to non-root user
USER user

# Expose standard HF Space / API port
EXPOSE 7860

# Healthcheck probe against the lightweight health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the FastAPI application via uvicorn
CMD ["sh", "-c", "uvicorn legalrag.api.main:app --host 0.0.0.0 --port ${API_PORT:-7860}"]
