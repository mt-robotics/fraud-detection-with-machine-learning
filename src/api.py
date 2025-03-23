"""Module for serving fraud detection model predictions via FastAPI."""

from datetime import datetime
from typing import Dict, Optional, Any  # , Union, List
import threading
import time
import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, field_validator, Field
import pandas as pd
import numpy as np
from joblib import load
from loguru import logger
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    start_http_server,
    # REGISTRY,
)

from config.config import MODELS_DIR, LOG_LEVEL, LOG_FORMAT


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types.

    Args:
        obj: Object potentially containing numpy types

    Returns:
        The same object with numpy types converted to Python native types
    """
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(x) for x in obj]
    else:
        return obj


# Configure logger
logger.remove()
logger.add("logs/api.log", level=LOG_LEVEL, format=LOG_FORMAT, rotation="500 MB")

# Configure Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP Requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP Request Duration in seconds",
    ["method", "endpoint"],
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds", "Model prediction duration in seconds"
)
FRAUD_RATIO = Gauge("fraud_ratio", "Ratio of transactions classified as fraud")
PREDICTION_COUNT = Counter("prediction_count", "Number of predictions made", ["result"])
MODEL_PREDICTION_SUMMARY = Summary(
    "model_prediction_summary", "Summary of model prediction times"
)

# Start metrics endpoint on a separate port
METRICS_PORT = int(os.environ.get("METRICS_PORT", 8001))


def start_metrics_server():
    """Start Prometheus metrics server on a separate thread."""
    start_http_server(METRICS_PORT)
    logger.info(f"Metrics server started on port {METRICS_PORT}")


# Start metrics server in a separate thread
threading.Thread(target=start_metrics_server, daemon=True).start()

app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting fraudulent transactions in digital payments",
    version="1.0.0",
)

# Model and metadata loading with error handling
try:
    # Load model
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load(model_path)

    # Load metadata (contains threshold, feature importance, etc.)
    metadata_path = MODELS_DIR / "model_metadata.joblib"
    if metadata_path.exists():
        metadata = load(metadata_path)
        threshold = metadata.get("threshold", 0.5)
        model_name = metadata.get("model_name", "unknown")
        top_features = metadata.get("top_features", [])
        # Ensure feature names are consistently lowercase
        feature_names = [name.lower() for name in metadata.get(
            "feature_names", [f"v{i}" for i in range(1, 29)] + ["time", "amount"]
        )]
    else:
        logger.warning("Model metadata not found, using default threshold and settings")
        threshold = 0.5
        model_name = "unknown"
        top_features = []
        feature_names = [f"v{i}" for i in range(1, 29)] + ["time", "amount"]  # Already lowercase

    logger.info(f"Successfully loaded {model_name} model from {model_path}")
    logger.info(f"Using classification threshold: {threshold}")
    if top_features:
        logger.info(f"Top predictive features: {', '.join(top_features[:5])}")

except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError("Application startup failed")


class TransactionData(BaseModel):
    """Pydantic model for transaction data validation."""

    # Allow both lowercase and uppercase field names to support both formats
    time: float = Field(..., description="Time in seconds since start of day", alias="Time")
    v1: float = Field(..., description="Anonymized PCA feature 1", alias="V1")
    v2: float = Field(..., description="Anonymized PCA feature 2", alias="V2")
    v3: float = Field(..., description="Anonymized PCA feature 3", alias="V3")
    v4: float = Field(..., description="Anonymized PCA feature 4", alias="V4")
    v5: float = Field(..., description="Anonymized PCA feature 5", alias="V5")
    v6: float = Field(..., description="Anonymized PCA feature 6", alias="V6")
    v7: float = Field(..., description="Anonymized PCA feature 7", alias="V7")
    v8: float = Field(..., description="Anonymized PCA feature 8", alias="V8")
    v9: float = Field(..., description="Anonymized PCA feature 9", alias="V9")
    v10: float = Field(..., description="Anonymized PCA feature 10", alias="V10")
    v11: float = Field(..., description="Anonymized PCA feature 11", alias="V11")
    v12: float = Field(..., description="Anonymized PCA feature 12", alias="V12")
    v13: float = Field(..., description="Anonymized PCA feature 13", alias="V13")
    v14: float = Field(..., description="Anonymized PCA feature 14", alias="V14")
    v15: float = Field(..., description="Anonymized PCA feature 15", alias="V15")
    v16: float = Field(..., description="Anonymized PCA feature 16", alias="V16")
    v17: float = Field(..., description="Anonymized PCA feature 17", alias="V17")
    v18: float = Field(..., description="Anonymized PCA feature 18", alias="V18")
    v19: float = Field(..., description="Anonymized PCA feature 19", alias="V19")
    v20: float = Field(..., description="Anonymized PCA feature 20", alias="V20")
    v21: float = Field(..., description="Anonymized PCA feature 21", alias="V21")
    v22: float = Field(..., description="Anonymized PCA feature 22", alias="V22")
    v23: float = Field(..., description="Anonymized PCA feature 23", alias="V23")
    v24: float = Field(..., description="Anonymized PCA feature 24", alias="V24")
    v25: float = Field(..., description="Anonymized PCA feature 25", alias="V25")
    v26: float = Field(..., description="Anonymized PCA feature 26", alias="V26")
    v27: float = Field(..., description="Anonymized PCA feature 27", alias="V27")
    v28: float = Field(..., description="Anonymized PCA feature 28", alias="V28")
    amount: float = Field(..., description="Transaction amount", alias="Amount")
    
    # Configure the model to work with both lowercase and uppercase names
    model_config = {
        "populate_by_name": True,  # Allow population by field name in addition to alias
        "extra": "ignore"          # Ignore any extra fields
    }

    # Using modern Pydantic v2 field_validator
    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be positive")
        return v

    @field_validator("time")
    @classmethod
    def time_must_be_valid(cls, v):
        if v < 0 or v > 86400:  # Seconds in a day
            raise ValueError("Time must be between 0 and 86400")
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    is_fraud: bool
    fraud_probability: float
    threshold_used: float
    status: str
    timestamp: str
    model_name: str
    explanation: Optional[Dict[str, float]] = None


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "model": model_name,
        "threshold": threshold,
        "model_path": str(model_path),
        "top_features": top_features[:5] if top_features else [],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionData,
    custom_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Custom classification threshold"
    ),
):
    """Endpoint for making fraud predictions.

    Args:
        transaction: Transaction data
        custom_threshold: Optional custom threshold (overrides default model threshold)

    Returns:
        Prediction response with fraud probability and classification
    """
    # Track request with Prometheus
    request_start = time.time()

    try:
        # Use custom threshold if provided, otherwise use model's default
        classification_threshold = (
            custom_threshold if custom_threshold is not None else threshold
        )

        # Convert input data to DataFrame using model_dump() (Pydantic v2 method)
        data_dict = transaction.model_dump()
        
        # Create a DataFrame from the data dictionary
        data = pd.DataFrame([data_dict])
        
        # Normalize column names to lowercase to match expected feature_names
        data.columns = [col.lower() for col in data.columns]
        
        # Ensure feature order matches what the model expects
        if feature_names and len(feature_names) == data.shape[1]:
            data = data[feature_names]
        elif feature_names:
            # If we have a mismatch, try to align the columns by name
            try:
                data = data.loc[:, [col.lower() for col in feature_names]]
                logger.info(f"Realigned columns to match feature_names")
            except KeyError as e:
                logger.error(f"Column alignment failed: {str(e)} - Available columns: {data.columns.tolist()}")
                raise ValueError(f"Input data columns do not match model's expected features: {str(e)}")

        # Log prediction request
        logger.info(
            f"Processing prediction request for transaction amount: {transaction.amount}, using threshold: {classification_threshold}"
        )

        # Make prediction
        start_time = datetime.now()
        try:
            # Safely extract and convert probability
            # Use a custom context for this operation to avoid MutexValue issues
            # This avoids the synchronization problems with MODEL_PREDICTION_SUMMARY.time()
            prediction_start = time.time()
            probabilities = model.predict_proba(data)
            prediction_duration = time.time() - prediction_start
            
            # Update metrics outside the model prediction
            MODEL_PREDICTION_SUMMARY.observe(prediction_duration)

            if probabilities is None or len(probabilities) == 0:
                raise ValueError("Model returned empty probability array")

            # Safely extract the probability value with proper error handling
            probability = convert_numpy_types(probabilities[0][1])
            
            # Convert to standard Python types to avoid any MutexValue issues
            probability = float(probability)
            classification_threshold = float(classification_threshold)
            
            is_fraud = probability >= classification_threshold
            prediction_time = (datetime.now() - start_time).total_seconds()

            # Update Prometheus metrics
            PREDICTION_LATENCY.observe(prediction_time)
            PREDICTION_COUNT.labels(result="fraud" if is_fraud else "legitimate").inc()

            # Track running fraud ratio
            # We use a gauge as an approximate measure since we can't track all-time totals directly
            try:
                # Use a safer approach for the fraud ratio that's resilient to concurrency issues
                if hasattr(FRAUD_RATIO, "_value") and FRAUD_RATIO._value is not None:
                    # Safely get the current value
                    try:
                        current_ratio = float(FRAUD_RATIO._value)
                        alpha = 0.05  # Decay factor for EMA
                        new_ratio = alpha * (1.0 if is_fraud else 0.0) + (1 - alpha) * current_ratio
                        FRAUD_RATIO.set(float(new_ratio))
                    except (TypeError, ValueError) as e:
                        # If there's any issue with the value, reset it
                        logger.warning(f"Error updating fraud ratio: {str(e)}")
                        FRAUD_RATIO.set(1.0 if is_fraud else 0.0)
                else:
                    # First observation
                    FRAUD_RATIO.set(1.0 if is_fraud else 0.0)
            except Exception as e:
                # Don't let metrics issues break the API functionality
                logger.warning(f"Metrics error: {str(e)}")

            logger.debug(f"Raw prediction probabilities: {probabilities}")
            logger.debug(
                f"Converted probability: {probability}, type: {type(probability)}"
            )

        except IndexError as e:
            logger.error(
                f"IndexError in probability extraction: {str(e)}, shape: {probabilities.shape if hasattr(probabilities, 'shape') else 'unknown'}"
            )
            raise ValueError(
                f"Failed to extract probability from model output: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")

        # Create response with properly converted types - ensure all values are standard Python types
        response = PredictionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=float(probability),
            threshold_used=float(classification_threshold),
            status="success",
            timestamp=pd.Timestamp.now().isoformat(),
            model_name=str(model_name),
        )

        # Add feature contribution information if top features is available
        if top_features:
            try:
                # Just include values of the top features with proper type conversion
                # and ensure they are Python native types, not MutexValue or numpy types
                explanation = {}
                for feature in top_features[:5]:
                    if feature in data.columns:
                        # Make sure to convert to native Python types
                        value = data[feature].iloc[0]
                        explanation[feature] = float(convert_numpy_types(value))
                    else:
                        # If feature not found, provide a safe default
                        explanation[feature] = 0.0
                
                response.explanation = explanation
            except Exception as e:
                # Don't let explanation generation break the prediction
                logger.warning(f"Error generating feature explanation: {str(e)}")
                # Provide empty explanation in case of error
                response.explanation = {}

        logger.info(
            f"Prediction complete in {prediction_time:.4f}s - Fraud probability: {probability:.4f}, classified as fraud: {is_fraud}"
        )

        # Update total request metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(
            time.time() - request_start
        )

        return response

    except Exception as e:
        # Update error metrics
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(
            time.time() - request_start
        )

        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "status": "error",
                "timestamp": pd.Timestamp.now().isoformat(),
            },
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    request_start = time.time()

    try:
        # Simple model sanity check
        test_data = pd.DataFrame(
            [[0] * 30], columns=["time"] + [f"v{i}" for i in range(1, 29)] + ["amount"]
        )
        # Ensure prediction results are properly converted
        prediction = convert_numpy_types(model.predict(test_data))

        # Create response with properly converted types
        response = {
            "status": "healthy",
            "model_loaded": True,
            "model_name": model_name,
            "prediction_test_result": prediction,
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Update metrics
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status="200").inc()
        REQUEST_LATENCY.labels(method="GET", endpoint="/health").observe(
            time.time() - request_start
        )

        return convert_numpy_types(response)
    except Exception as e:
        # Update error metrics
        REQUEST_COUNT.labels(method="GET", endpoint="/health", status="500").inc()
        REQUEST_LATENCY.labels(method="GET", endpoint="/health").observe(
            time.time() - request_start
        )

        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get information about the deployed model."""
    request_start = time.time()

    try:
        # Fix the reference to metadata by using the properly loaded variable
        model_metrics = {}
        training_date = "unknown"

        # Use the metadata variable that was loaded earlier
        if "metadata" in locals() or "metadata" in globals():
            if isinstance(metadata, dict):
                model_metrics = metadata.get("metrics", {})
                # Use the utility function to convert numpy types
                model_metrics = convert_numpy_types(model_metrics)

                training_date = metadata.get("training_date", "unknown")

        # Build response with consistent type conversion
        response = {
            "model_name": model_name,
            "threshold": threshold,  # Will be converted below
            "top_features": top_features[:5] if top_features else [],
            "metrics": model_metrics,
            "training_date": training_date,
        }

        # Update metrics
        REQUEST_COUNT.labels(method="GET", endpoint="/model/info", status="200").inc()
        REQUEST_LATENCY.labels(method="GET", endpoint="/model/info").observe(
            time.time() - request_start
        )

        # Convert any remaining numpy types in the entire response
        return convert_numpy_types(response)
    except Exception as e:
        # Update error metrics
        REQUEST_COUNT.labels(method="GET", endpoint="/model/info", status="500").inc()
        REQUEST_LATENCY.labels(method="GET", endpoint="/model/info").observe(
            time.time() - request_start
        )

        logger.error(f"Error retrieving model info: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve model info: {str(e)}"
        )


@app.get("/model/threshold")
async def get_threshold_info():
    """Get information about the model's classification threshold."""
    request_start = time.time()

    response = {
        "current_threshold": threshold,
        "description": "Classification threshold for converting fraud probabilities to binary predictions",
        "note": "You can provide a custom threshold using the custom_threshold query parameter in the /predict endpoint",
    }

    # Update metrics
    REQUEST_COUNT.labels(method="GET", endpoint="/model/threshold", status="200").inc()
    REQUEST_LATENCY.labels(method="GET", endpoint="/model/threshold").observe(
        time.time() - request_start
    )

    # Convert any numpy types in the response
    return convert_numpy_types(response)


@app.get("/metrics")
async def metrics_redirect():
    """Redirect to the Prometheus metrics endpoint."""
    return {
        "message": "Prometheus metrics are available on a separate port",
        "metrics_url": f"http://localhost:{METRICS_PORT}/metrics",
        "note": "This endpoint is for informational purposes only. Use the dedicated metrics port for Prometheus scraping.",
    }
