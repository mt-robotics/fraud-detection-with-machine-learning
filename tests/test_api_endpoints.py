"""Tests for API endpoints with mock model."""

import sys
from pathlib import Path

# from unittest.mock import patch
# import pytest
from fastapi.testclient import TestClient
import numpy as np

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def generate_test_transaction(fraud_like=False, seed=42):
    """Generate test transaction data with Python native types."""
    np.random.seed(seed)

    # Helper function to convert numpy values to native Python types
    def to_python_float(value):
        """Convert numpy float to Python float."""
        return float(value)

    if fraud_like:
        # Generate transaction with features similar to fraud cases
        return {
            "time": to_python_float(np.random.uniform(0, 86400)),
            "v1": to_python_float(
                np.random.uniform(-3, -1)
            ),  # Negative values more common in fraud
            "v2": to_python_float(np.random.uniform(-3, -1)),
            "v3": to_python_float(np.random.uniform(3, 5)),
            "v4": to_python_float(np.random.uniform(-5, -2)),
            "v5": to_python_float(np.random.uniform(-5, -2)),
            "v6": to_python_float(np.random.uniform(0, 3)),
            "v7": to_python_float(np.random.uniform(0, 3)),
            "v8": to_python_float(np.random.uniform(-2, 2)),
            "v9": to_python_float(np.random.uniform(-2, 2)),
            "v10": to_python_float(np.random.uniform(-3, -1)),
            "v11": to_python_float(np.random.uniform(0, 3)),
            "v12": to_python_float(np.random.uniform(-3, 3)),
            "v13": to_python_float(np.random.uniform(-3, 3)),
            "v14": to_python_float(np.random.uniform(-5, -2)),
            "v15": to_python_float(np.random.uniform(1, 5)),
            "v16": to_python_float(np.random.uniform(-2, 2)),
            "v17": to_python_float(np.random.uniform(-2, 2)),
            "v18": to_python_float(np.random.uniform(-2, 2)),
            "v19": to_python_float(np.random.uniform(-2, 2)),
            "v20": to_python_float(np.random.uniform(-2, 2)),
            "v21": to_python_float(np.random.uniform(-2, 2)),
            "v22": to_python_float(np.random.uniform(-2, 2)),
            "v23": to_python_float(np.random.uniform(-2, 2)),
            "v24": to_python_float(np.random.uniform(-2, 2)),
            "v25": to_python_float(np.random.uniform(-2, 2)),
            "v26": to_python_float(np.random.uniform(-2, 2)),
            "v27": to_python_float(np.random.uniform(-2, 2)),
            "v28": to_python_float(np.random.uniform(-2, 2)),
            "amount": to_python_float(
                np.random.uniform(1, 2500)
            ),  # Higher amounts more common in fraud
        }
    else:
        # Generate normal transaction
        return {
            "time": to_python_float(np.random.uniform(0, 86400)),
            "v1": to_python_float(np.random.uniform(-1, 3)),
            "v2": to_python_float(np.random.uniform(-1, 3)),
            "v3": to_python_float(np.random.uniform(-3, 1)),
            "v4": to_python_float(np.random.uniform(-1, 3)),
            "v5": to_python_float(np.random.uniform(-1, 3)),
            "v6": to_python_float(np.random.uniform(-3, 1)),
            "v7": to_python_float(np.random.uniform(-3, 1)),
            "v8": to_python_float(np.random.uniform(-2, 2)),
            "v9": to_python_float(np.random.uniform(-2, 2)),
            "v10": to_python_float(np.random.uniform(-1, 3)),
            "v11": to_python_float(np.random.uniform(-3, 1)),
            "v12": to_python_float(np.random.uniform(-3, 3)),
            "v13": to_python_float(np.random.uniform(-3, 3)),
            "v14": to_python_float(np.random.uniform(-1, 3)),
            "v15": to_python_float(np.random.uniform(-3, 1)),
            "v16": to_python_float(np.random.uniform(-2, 2)),
            "v17": to_python_float(np.random.uniform(-2, 2)),
            "v18": to_python_float(np.random.uniform(-2, 2)),
            "v19": to_python_float(np.random.uniform(-2, 2)),
            "v20": to_python_float(np.random.uniform(-2, 2)),
            "v21": to_python_float(np.random.uniform(-2, 2)),
            "v22": to_python_float(np.random.uniform(-2, 2)),
            "v23": to_python_float(np.random.uniform(-2, 2)),
            "v24": to_python_float(np.random.uniform(-2, 2)),
            "v25": to_python_float(np.random.uniform(-2, 2)),
            "v26": to_python_float(np.random.uniform(-2, 2)),
            "v27": to_python_float(np.random.uniform(-2, 2)),
            "v28": to_python_float(np.random.uniform(-2, 2)),
            "amount": to_python_float(
                np.random.uniform(1, 500)
            ),  # Lower amounts more common in normal
        }


def test_predict_endpoint_normal(app_with_mock_model, mock_model):
    """Test prediction endpoint with normal transaction."""
    # Configure the mock to return low fraud probability
    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])  # 5% fraud prob
    client = TestClient(app_with_mock_model)

    # Test the endpoint
    transaction = generate_test_transaction(fraud_like=False)
    response = client.post("/predict", json=transaction)

    # Check the response
    assert response.status_code == 200, f"API error: {response.text}"
    result = response.json()
    assert result["fraud_probability"] == 0.05
    assert "is_fraud" in result
    assert (
        result["is_fraud"] is False
    )  # Should be False since 0.05 < 0.5 (default threshold)
    assert "threshold_used" in result
    assert result["threshold_used"] == 0.5  # Check default threshold
    assert "status" in result
    assert result["status"] == "success"
    assert "model_name" in result
    assert "timestamp" in result


def test_predict_endpoint_fraud(app_with_mock_model, mock_model):
    """Test prediction endpoint with fraud transaction."""
    # Configure the mock to return high fraud probability
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% fraud prob
    client = TestClient(app_with_mock_model)

    # Test the endpoint
    transaction = generate_test_transaction(fraud_like=True)
    response = client.post("/predict", json=transaction)

    # Check the response
    assert response.status_code == 200, f"API error: {response.text}"
    result = response.json()
    assert result["fraud_probability"] == 0.8
    assert "is_fraud" in result
    assert (
        result["is_fraud"] is True
    )  # Should be True since 0.8 > 0.5 (default threshold)
    assert "threshold_used" in result
    assert result["threshold_used"] == 0.5  # Check default threshold
    assert "status" in result
    assert result["status"] == "success"
    assert "model_name" in result
    assert "timestamp" in result


def test_predict_with_custom_threshold(app_with_mock_model, mock_model):
    """Test prediction with a custom threshold."""
    # Configure the mock to return medium fraud probability
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])  # 30% fraud prob
    client = TestClient(app_with_mock_model)

    # Set a custom threshold
    custom_threshold = 0.2  # Lower than default

    # Test the endpoint
    transaction = generate_test_transaction()
    response = client.post(
        f"/predict?custom_threshold={custom_threshold}", json=transaction
    )

    # Check the response
    assert response.status_code == 200, f"API error: {response.text}"
    result = response.json()
    assert result["fraud_probability"] == 0.3
    assert result["threshold_used"] == custom_threshold
    assert result["is_fraud"] is True  # Since 0.3 > 0.2
    assert "status" in result
    assert result["status"] == "success"
    assert "model_name" in result
    assert "timestamp" in result


def test_health_check(app_with_mock_model, mock_model):
    """Test health check endpoint."""
    # Configure the mock for health check
    mock_model.predict.return_value = np.array([0])  # Non-fraud prediction
    client = TestClient(app_with_mock_model)

    # Test the endpoint
    response = client.get("/health")

    # Check the response
    assert response.status_code == 200, f"API error: {response.text}"
    result = response.json()
    assert result["status"] == "healthy"
    assert result["model_loaded"] is True
    assert "model_name" in result
    assert "timestamp" in result
    assert "prediction_test_result" in result


def test_model_info(app_with_mock_model):
    """Test model info endpoint."""
    client = TestClient(app_with_mock_model)

    # Test the endpoint
    response = client.get("/model/info")

    # Check the response
    assert response.status_code == 200, f"API error: {response.text}"
    result = response.json()
    assert "model_name" in result
    assert "threshold" in result
    assert "metrics" in result
    assert "top_features" in result
    assert "training_date" in result
