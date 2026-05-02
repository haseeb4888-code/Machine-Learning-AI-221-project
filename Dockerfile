# ============================================================
# Economic Growth Analyzer — Dockerfile (Fixed)
# Matches actual repo: endpoints.py + models/ + CSV dataset
# ============================================================

# ── Stage 1: Builder (install heavy deps) ────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies (cached as its own layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production (lean final image) ───────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Only runtime libs needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# ── Copy your actual project files ───────────────────────────
COPY ["countries of the world.csv", "./countries of the world.csv"]
COPY models/ ./models/
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY results/ ./results/

# ── Security: non-root user ───────────────────────────────────
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────────────────────
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD sh -c 'curl -f http://localhost:${PORT:-8000}/health || exit 1'

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
