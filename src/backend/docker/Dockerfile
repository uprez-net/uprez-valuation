# Multi-stage Dockerfile for IPO Valuation Platform Backend

# Base stage with common dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    libssl-dev \
    pkg-config \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -d /app -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/requirements-dev.txt ./requirements-dev.txt
COPY requirements/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Development command
CMD ["uvicorn", "src.backend.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy requirements
COPY requirements/requirements.txt ./requirements.txt

# Install Python dependencies (production only)
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Copy source code
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["gunicorn", "src.backend.api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]

# Celery worker stage
FROM production as celery-worker

# Celery command
CMD ["celery", "-A", "src.backend.messaging.celery_app", "worker", "--loglevel=info", "--concurrency=2"]

# Celery beat stage (scheduler)
FROM production as celery-beat

# Celery beat command
CMD ["celery", "-A", "src.backend.messaging.celery_app", "beat", "--loglevel=info"]

# ML service stage (with additional ML libraries)
FROM base as ml-service

# Install additional ML dependencies
RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Copy ML-specific requirements
COPY requirements/requirements.txt ./requirements.txt

# Install Python dependencies with optimizations for ML
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ ./src/

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8001

# ML service command
CMD ["uvicorn", "src.backend.ml_services.main:app", "--host", "0.0.0.0", "--port", "8001"]