# Multi-stage build for production optimization
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY src/realtime/config/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --system --gid=1001 uprez && \
    useradd --system --uid=1001 --gid=uprez --create-home --shell=/bin/bash uprez

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=uprez:uprez src/ ./src/
COPY --chown=uprez:uprez pyproject.toml ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data && chown -R uprez:uprez /app

# Switch to non-root user
USER uprez

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.realtime.examples.basic_usage:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# Development stage
FROM production as development

# Install development dependencies
USER root
COPY src/realtime/config/requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install development tools
RUN apt-get update && apt-get install -y \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER uprez

# Override command for development
CMD ["uvicorn", "src.realtime.examples.basic_usage:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]