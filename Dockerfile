# ── Multi-stage Dockerfile for Healthcare AI Assistant ───────────────────────
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer outputs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=.

# Install system dependencies (lxml C-bindings, curl, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libxml2-dev \
    libxslt1-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy workspace project files
COPY . .

# Grant execution permissions to entrypoint script
RUN chmod +x entrypoint.sh

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit application status
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Execute single-click pipeline entrypoint
ENTRYPOINT ["./entrypoint.sh"]
