FROM python:3.11-slim

WORKDIR /app

# Set build-time environment variables (these don't change between environments)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Note: ENV variable should be provided at runtime, not hardcoded here
# Default will be handled by docker-compose or docker-entrypoint.sh

# Note: Runtime environment variables like API_HOST, API_PORT, LOG_LEVEL
# will be provided by docker-compose or docker run from the appropriate .env.{ENV} file
# This allows different values in different environments without rebuilding the image

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p data/raw data/processed data/processed/baseline models logs && \
    chmod +x scripts/train_and_evaluate.sh

# Add metadata
LABEL maintainer="AI Team" \
      version="2.0" \
      description="Enhanced Fraud Detection API with ML ensemble models"

# Create volumes for persistent storage
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Expose the port the app runs on (default, can be overridden at runtime)
EXPOSE 8000 8001

# Healthcheck with more realistic parameters for ML model loading
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Add startup script to prepare environment
COPY scripts/docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint script for initialization
ENTRYPOINT ["docker-entrypoint.sh"]

# Command to run the application using environment variables
CMD uvicorn src.api:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000} --workers ${API_WORKERS:-2} --log-level ${LOG_LEVEL:-info}
