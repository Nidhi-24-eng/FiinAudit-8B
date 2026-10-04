# ─────────────────────────────────────────────────────────────────────────────
# FinAudit-8B — Multi-Stage Dockerfile (Day 9)
# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Build — install Python dependencies
# Stage 2: Runtime — minimal CUDA runtime image
#
# Build:  docker build -t finaudit-8b:latest .
# Run:    docker compose up
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Dependency builder ───────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast package installer)
RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install all dependencies into a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache-dir \
        torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 && \
    uv pip install --no-cache-dir vllm autoawq && \
    uv pip install --no-cache-dir fastapi uvicorn[standard] pydantic sse-starlette \
        deepeval pytest httpx python-dotenv rich typer


# ── Stage 2: Production runtime ───────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Minimal runtime OS packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application source
COPY app/       ./app/
COPY src/       ./src/
COPY benchmarks/ ./benchmarks/
COPY tests/     ./tests/

# Default env — override via docker-compose or -e flags
ENV MODEL_PATH=/models/finaudit-awq
ENV MAX_MODEL_LEN=4096
ENV GPU_MEM_UTILIZATION=0.90
ENV MAX_NUM_SEQS=64
ENV TENSOR_PARALLEL=1

# Expose API port
EXPOSE 8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
