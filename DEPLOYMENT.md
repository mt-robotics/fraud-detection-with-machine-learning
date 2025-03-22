# Fraud Detection API Deployment Guide

This document provides instructions for deploying the Fraud Detection API using Docker.

## Prerequisites

- Docker and Docker Compose installed
- Git (for cloning the repository)

## Quick Start

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd fraud_detection_with_machine_learning
   ```

2. Create an `.env` file with the following environment variables:

   ```
   ENV=development # or staging, production
   ```

3. Create an environment-specific `.env` file, for example `.env.development`, with the following environment variables:

   ```
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   LOG_LEVEL=<desired-log-level> # e.g., "INFO", "WARNING", "ERROR"

   # Model Configuration
   ADVANCED_FEATURES=<true|false>
   ENSEMBLE_MODELS=<true|false>
   MODEL_VERSION=<model-version>

   # Metrics Configuration
   METRICS_PORT=8001

   # Docker Configuration
   DOCKER_REGISTRY=<myregistry.example.com>
   DOCKER_TAG=<desired-tag>

   # Set this to true to run tests on container startup
   # RUN_TESTS=<true|false>
   ```

4. Launch the API service:

   ```bash
   docker-compose up -d
   ```

5. To include monitoring tools (Prometheus and Grafana):
   ```bash
   docker-compose --profile monitoring up -d
   ```

## Deployment Options

### Standard Deployment

The standard deployment includes just the Fraud Detection API:

```bash
docker-compose up -d
```

This will:

- Build the Docker image from the Dockerfile
- Start the API container with appropriate volumes for data persistence
- Map port 8000 to your host machine

### Monitoring Deployment

For a deployment with monitoring capabilities:

```bash
docker-compose --profile monitoring up -d
```

This adds:

- Prometheus for metrics collection (accessible at http://localhost:9090)
- Grafana for visualization (accessible at http://localhost:3000, default login: admin/admin)

## Volume Management

The following volumes are maintained:

- `./data:/app/data`: For storing raw and processed datasets
- `./models:/app/models`: For storing trained ML models
- `./logs:/app/logs`: For API and application logs

To ensure these directories have appropriate permissions:

```bash
mkdir -p data/raw data/processed models logs
chmod 777 data models logs  # Ensure Docker can write to these directories
```

## Environment Variables

| Variable          | Description                         | Default                |
| ----------------- | ----------------------------------- | ---------------------- |
| API_HOST          | Host address for the API            | 0.0.0.0                |
| API_PORT          | Port for the API                    | 8000                   |
| LOG_LEVEL         | Logging level                       | INFO                   |
| ADVANCED_FEATURES | Enable advanced feature engineering | true                   |
| ENSEMBLE_MODELS   | Enable ensemble model training      | true                   |
| MODEL_VERSION     | Version of the trained model        | 1.0.0                  |
| METRICS_PORT      | Port for Prometheus metrics         | 8001                   |
| DOCKER_REGISTRY   | Docker registry for images          | myregistry.example.com |
| DOCKER_TAG        | Tag for Docker images               | latest                 |
| RUN_TESTS         | Run test suite on startup           | false                  |

## Health Checks

The API includes a health endpoint at `/health` that can be used to monitor its status. The Docker container is configured with health checks that will report container health status based on this endpoint.

## Scaling

To scale the API horizontally (must configure a load balancer separately):

```bash
docker-compose up -d --scale fraud-detection-api=3
```

## Troubleshooting

### View Logs

```bash
# API logs
docker-compose logs -f fraud-detection-api

# All services logs
docker-compose logs -f
```

### Check Container Status

```bash
docker-compose ps
```

### Restart Services

```bash
docker-compose restart fraud-detection-api
```

### Common Issues

1. **Model not found**:

   - Ensure you have mounted the correct volume for models
   - The container will attempt to train a new model if the dataset is available

2. **Container crashes on startup**:

   - Check logs with `docker-compose logs fraud-detection-api`
   - Ensure memory limits are sufficient for model loading

3. **Performance issues**:
   - Adjust the CPU and memory limits in docker-compose.yml
   - Consider scaling horizontally with multiple replicas

## CI/CD Integration

This project includes GitHub Actions workflows for Continuous Integration and Deployment. See `.github/workflows/ci-cd.yml` for details.

## Security Considerations

- The default Grafana credentials should be changed in production
- Consider using Docker secrets for sensitive environment variables
- Implement proper network segmentation in production environments
- Set up proper authentication for the API endpoints in production
