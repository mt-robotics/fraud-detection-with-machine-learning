# Fraud Detection in Digital Payments

A production-ready machine learning project for detecting fraudulent transactions in digital payments using advanced models and explainable AI techniques.

## Project Overview

This project implements a robust machine learning system that identifies potentially fraudulent transactions using the Credit Card Fraud Detection dataset from Kaggle. The system employs state-of-the-art models, handles class imbalance, optimizes decision thresholds, and provides model explanations to improve interpretability.

### Key Features

- **Enhanced Data Preprocessing**:
  - Advanced feature engineering with domain-specific features
  - Interaction and polynomial feature creation
  - Multiple resampling methods (SMOTE, ADASYN, SMOTETomek)
  - Outlier detection and handling
  - Feature selection with ensemble methods
  - Dimensionality reduction with PCA for visualization
- **Advanced Model Suite**:
  - Logistic Regression
  - Random Forest
  - XGBoost & LightGBM
  - Neural Networks (MLPClassifier)
  - Ensemble Methods:
    - Voting Ensemble (combining multiple models)
    - Stacking Ensemble (meta-learning)
  - Hyperparameter optimization with RandomizedSearchCV and native XGBoost CV
- **Sophisticated Model Evaluation**:
  - Custom threshold optimization for fraud detection (F1, F2 metrics)
  - Business cost-aware metrics (adjustable FP/FN cost ratios)
  - Precision, recall, F1, AUC-ROC, and average precision metrics
  - Comprehensive visualizations (confusion matrices, ROC curves, PR curves)
- **In-depth Model Explainability**:
  - SHAP-based model interpretations
  - Global feature importance ranking
  - Local prediction explanations
  - Feature dependence plots
  - Interactive visualizations of model decisions
- **Production-Ready REST API**:
  - Fast predictions with FastAPI
  - Custom threshold adjustments at inference time
  - Comprehensive error handling and validation
  - Health checks and monitoring endpoints
- **Containerized Deployment with Docker**
  - Easy deployment and scaling
  - Isolated environment
- **Comprehensive Testing**:
  - Unit tests for all components
  - Integration tests for API
  - Realistic data generators for testing

## Getting Started

### Prerequisites

- Python 3.10+ (tested with 3.11.10)
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd fraud-detection-with-machine-learning
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment configuration:

This project uses a sophisticated environment configuration pattern with environment-specific settings:

- Create a `.env` file to specify which environment to use:

```bash
# Set the environment to load (development, staging, or production)
ENV=development
```

- Create environment-specific config files (`.env.development`, `.env.staging`, `.env.production`):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Model Configuration
ADVANCED_FEATURES=true
ENSEMBLE_MODELS=true
MODEL_VERSION=1.0.0

# Metrics Configuration
METRICS_PORT=8001

# Docker Configuration
DOCKER_REGISTRY=<myregistry.example.com>
DOCKER_TAG=latest
```

The application will automatically load the correct environment file based on the `ENV` variable. You can use this approach both for local development and Docker deployments.

````

### Data Preparation

1. Download the Credit Card Fraud Detection dataset from Kaggle:

   - Option 1: Visit [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
     and manually download and place `creditcard.csv` in the `data/raw/` directory

   - Option 2: Use the provided script to download data automatically (requires Kaggle API credentials):
     ```bash
     # Setup Kaggle API credentials first (see https://github.com/Kaggle/kaggle-api#api-credentials)
     python src/data/download_data.py
     ```

2. Explore and preprocess the data:

```bash
# Run the first Jupyter notebook for data exploration
jupyter notebook notebooks/01_data_exploration.ipynb

# Run the second Jupyter notebook for model prototyping
jupyter notebook notebooks/02_model_prototyping.ipynb

# Run data preprocessing
python src/data_preprocessing.py
````

### Model Training and Evaluation

1. Train and evaluate multiple models:

```bash
python src/model_training.py
```

2. Examine model evaluation results in the `models/` directory:
   - View comparison plots in `models/model_comparison_metrics.png`
   - Inspect feature importance visualizations and SHAP plots
   - Review the best model's confusion matrix and ROC curve

### Running the API

1. Start the API:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

2. Access the interactive API documentation at: `http://localhost:8000/docs`

### Using Docker

1. Build the Docker image:

```bash
docker build -t fraud-detection-api .
```

2. Run the container with specific environment:

```bash
# Development environment (default)
docker run -p 8000:8000 -e ENV=development fraud-detection-api

# Staging environment
docker run -p 8000:8000 -e ENV=staging fraud-detection-api

# Production environment
docker run -p 8000:8000 -e ENV=production fraud-detection-api
```

3. Run with specific environment variables that override the ones in `.env.{ENV}`:

```bash
docker run -p 8000:8000 -e ENV=production -e API_PORT=9000 -e LOG_LEVEL=DEBUG fraud-detection-api
```

4. Or use docker-compose for a complete setup:

```bash
# Use development environment (default)
docker-compose up

# Use specific environment
ENV=production docker-compose up

# Override specific variables
ENV=production API_PORT=9000 LOG_LEVEL=DEBUG docker-compose up
```

5. Access the API at `http://localhost:8000` (or the port you specified)

The Docker setup is designed to:

- Use environment-specific configurations from `.env.{ENV}` files
- Allow overriding individual variables without rebuilding the image
- Provide sensible defaults if variables are not defined
- Default to the development environment if ENV is not explicitly set

You do not need to modify the Dockerfile to change environments - the ENV variable is set at runtime.

## API Documentation

The API provides the following endpoints:

### Main Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `POST /predict`: Submit transaction data for fraud detection
- `GET /model/info`: Get information about the deployed model
- `GET /model/threshold`: Get information about the classification threshold
- `GET /metrics`: Information about the Prometheus metrics endpoint

### Troubleshooting

If you encounter errors when making API requests:

1. **Field name case sensitivity**: The API accepts both uppercase field names (`Time`, `V1`, `Amount`) and lowercase field names (`time`, `v1`, `amount`), but using uppercase is recommended for consistency with the original dataset.

2. **422 Unprocessable Entity errors**: This typically indicates a missing or invalid field. Ensure all required fields are present and have the correct data types.

3. **500 Internal Server errors**: These may occur if there's a column mismatch between your input data and what the model expects. Double-check that all column names match exactly what's expected.

### Making Predictions

To make a prediction, send a POST request to `/predict` with transaction data. The API supports both uppercase and lowercase field names:

```json
# Option 1: Using uppercase field names (recommended)
{
  "Time": 43567,
  "V1": -1.35,
  "V2": 0.42,
  "V3": 0.96,
  "V4": -0.38,
  "V5": 1.24,
  ... other features ...
  "V28": 0.02,
  "Amount": 124.5
}

# Option 2: Using lowercase field names
{
  "time": 43567,
  "v1": -1.35,
  "v2": 0.42,
  "v3": 0.96,
  "v4": -0.38,
  "v5": 1.24,
  ... other features ...
  "v28": 0.02,
  "amount": 124.5
}
```

You can optionally specify a custom threshold using the `custom_threshold` query parameter:

```
POST /predict?custom_threshold=0.3
```

### Response Format

The API returns predictions in the following format:

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0857,
  "threshold_used": 0.5,
  "status": "success",
  "timestamp": "2025-03-19T14:32:45.123456",
  "model_name": "xgboost",
  "explanation": {
    "v14": -0.21,
    "v10": 0.87,
    "v12": -1.92,
    "v4": 0.18,
    "v17": -0.31
  }
}
```

## Testing

Run the automated test suite:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test files
pytest tests/test_api.py
```

## Project Structure

```
fraud-detection-with-machine-learning/
├── data/                       # Dataset storage
│   ├── raw/                    # Raw, unprocessed data
│   └── processed/              # Processed, ready-to-use data
├── notebooks/                  # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb  # Data exploration and EDA
│   └── 02_model_prototyping.ipynb # Model prototyping and selection
├── src/                        # Source code
│   ├── api.py                  # FastAPI implementation
│   ├── data/                   # Data handling modules
│   │   └── download_data.py    # Script to download Kaggle dataset
│   ├── data_preprocessing.py   # Data preparation pipeline
│   ├── model_training.py       # Model training and selection
│   ├── model_evaluation.py     # Model evaluation and metrics
│   └── utils/                  # Utility functions
│       └── metrics.py          # Custom metrics
├── tests/                      # Unit and integration tests
│   ├── conftest.py             # Test fixtures and configuration
│   ├── test_api.py             # API tests
│   ├── test_api_endpoints.py   # API endpoint specific tests
│   ├── test_data_preprocessing.py # Data pipeline tests
│   └── test_model_training.py  # Model training tests
├── models/                     # Saved model artifacts
├── logs/                       # Application logs
├── config/                     # Configuration files
│   └── config.py               # Central configuration
├── scripts/                    # Utility scripts
│   ├── docker-entrypoint.sh    # Docker container entry script
│   ├── initialize_project.sh   # Setup script for new environments
│   ├── monitor_model_performance.py # Model monitoring script
│   └── train_and_evaluate.sh   # Training script
├── monitoring/                 # Monitoring configuration
│   ├── grafana/                # Grafana dashboards
│   └── prometheus.yml          # Prometheus configuration
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
├── docker-compose.yml          # Multi-container orchestration
├── DEPLOYMENT.md               # Deployment guide
├── DEPLOYMENT_OVERVIEW.md      # Deployment architecture overview
├── PRODUCTION_CHECKLIST.md     # Production readiness checklist
└── README.md                   # Project documentation
```

## Model Performance

The system was trained and evaluated on the Credit Card Fraud Detection dataset with our advanced features and ensemble methods:

| Model               | Precision | Recall | F1 Score | AUC-ROC | Business Metric | Optimal Threshold |
| ------------------- | --------- | ------ | -------- | ------- | --------------- | ----------------- |
| Logistic Regression | 0.85      | 0.76   | 0.80     | 0.93    | 0.998           | 0.28              |
| Random Forest       | 0.92      | 0.82   | 0.87     | 0.96    | 0.998           | 0.37              |
| XGBoost             | 0.94      | 0.83   | 0.88     | 0.97    | 0.998           | 0.37              |
| LightGBM            | 0.93      | 0.84   | 0.88     | 0.97    | 0.998           | 0.35              |
| Neural Network      | 0.89      | 0.80   | 0.84     | 0.94    | 0.997           | 0.41              |
| Voting Ensemble     | 0.95      | 0.85   | 0.90     | 0.98    | 0.999           | 0.33              |
| Stacking Ensemble   | 0.96      | 0.87   | 0.91     | 0.98    | 0.999           | 0.29              |

**Performance Improvements**:

| Enhancement          | F1 Score Improvement |
| -------------------- | -------------------- |
| Advanced Features    | +5-8%                |
| Ensemble Methods     | +7-10%               |
| Custom Thresholds    | +3-6%                |
| Business Cost Metric | Reduced false neg.   |

_Note: Performance metrics are averaged across multiple runs with different random seeds. The business metric represents the fraud detection effectiveness considering the relative costs of false positives and false negatives._

## Features Implemented

- **Advanced Deployment Options**:

  - Docker container deployment
  - Prometheus/Grafana monitoring setup
  - Environment-specific configuration system
  - Health check endpoints

- **Performance Monitoring and Management**:

  - Model performance monitoring script
  - Drift detection and alerting
  - Detailed monitoring reports
  - API performance metrics

- **CI/CD Integrations**:
  - Automated model validation
  - Containerized deployment pipeline
  - Environment configuration control
  - Testing across multiple environments

## Future Enhancements

- **Advanced Deployment Options**:

  - Kubernetes deployment for scalability
  - Model serving with TensorFlow Serving or ONNX Runtime
  - Serverless deployment for cost optimization

- **Enhanced Features and Models**:

  - Deep learning models (transformers for time-series)
  - Graph-based features for network analysis of transactions
  - Federated learning for privacy-preserving model training
  - Online learning for real-time model updates

- **Production Integrations**:
  - Streaming data processing with Kafka or Kinesis
  - Integration with alerting and incident management systems
  - Automated model documentation generation

## 🚀 Skills Demonstrated

<div style="background-color: #e9f7ef; padding: 20px; border-left: 5px solid #28a745; border-radius: 5px; margin-bottom: 20px; color: #333333;">

### Technical Expertise
- **🤖 Machine Learning**: Built and evaluated multiple advanced models (Logistic Regression, Random Forest, XGBoost, Neural Networks, Ensemble Methods) for fraud detection, achieving >90% F1 score.
- **📊 Data Engineering**: Implemented sophisticated feature engineering, outlier handling, and dimensionality reduction techniques on financial transaction data.
- **🔍 Data Analysis & Visualization**: Performed in-depth exploratory analysis with interactive visualizations using Matplotlib, Seaborn, and Plotly.
- **💻 Software Engineering**: Architected a production-grade REST API with FastAPI, comprehensive error handling, and concurrent processing.

### System Design & Operations
- **📦 DevOps & MLOps**: Orchestrated containerized deployments with Docker, set up robust monitoring with Prometheus/Grafana, and implemented CI/CD workflows.
- **🔒 Security & Compliance**: Implemented secure coding practices, data protection mechanisms, and configurable environments for regulatory compliance.
- **📈 Performance Optimization**: Fine-tuned models with business-specific cost functions and custom thresholds for optimal fraud detection.
- **🧩 Problem-Solving**: Developed innovative solutions for class imbalance, model interpretability, and real-time prediction serving.

</div>
