"""Tests for the fraud detection API."""

import sys
from pathlib import Path
from typing import Dict
from fastapi.testclient import TestClient
import numpy as np
from unittest.mock import patch, MagicMock


# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.api import app  # pylint: disable=wrong-import-position

client = TestClient(app)


def generate_test_transaction(
    fraud_like: bool = False, seed: int = 42
) -> Dict[str, float]:
    """Generate test transaction data with Python native types.

    Args:
        fraud_like: Whether to generate transaction with fraud-like features
        seed: Random seed for reproducibility

    Returns:
        Dictionary with transaction features
    """
    np.random.seed(seed)

    # Helper function to convert numpy values to Python native types
    def to_python_float(value):
        """Convert numpy float to Python float."""
        return float(value)

    if fraud_like:
        # Generate transaction with features similar to fraud cases
        # Note: This is simplified and not based on actual fraud patterns in the data
        # In a real project, you would derive these from analysis of fraud examples
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


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "model" in response.json()
    assert "threshold" in response.json()


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "model_name" in response.json()


def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert "model_name" in response.json()
    assert "threshold" in response.json()
    assert "top_features" in response.json()


def test_threshold_info():
    """Test threshold info endpoint."""
    response = client.get("/model/threshold")
    assert response.status_code == 200
    assert "current_threshold" in response.json()
    assert "description" in response.json()


def test_predict_normal_transaction():
    """Test prediction endpoint with normal transaction."""
    test_transaction = generate_test_transaction(fraud_like=False)

    # Set up the mock data and model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.95, 0.05]]
    )  # 5% fraud probability

    # Mock metadata values
    mock_metadata = {
        "threshold": 0.5,
        "model_name": "Test XGBoost Model",
        "top_features": ["v14", "v4", "v12", "v10", "v11"],
        "metrics": {"accuracy": 0.95, "f1": 0.85},
    }

    # Create patches for all the necessary components
    with patch("src.api.model", mock_model), patch(
        "src.api.threshold", mock_metadata["threshold"]
    ), patch("src.api.model_name", mock_metadata["model_name"]), patch(
        "src.api.top_features", mock_metadata["top_features"]
    ), patch(
        "src.api.metadata", mock_metadata
    ), patch(
        "src.api.feature_names", [f"v{i}" for i in range(1, 29)] + ["time", "amount"]
    ):

        # Create a new TestClient for the patched app
        client = TestClient(app)  # pylint: disable=redefined-outer-name

        # Make the request
        response = client.post("/predict", json=test_transaction)

        # Check the response
        assert response.status_code == 200, f"API error: {response.text}"
        result = response.json()

        # Verify the response contains all expected fields
        assert "is_fraud" in result
        assert result["is_fraud"] is False  # 0.05 < 0.5 threshold
        assert "fraud_probability" in result
        assert result["fraud_probability"] == 0.05
        assert "threshold_used" in result
        assert result["threshold_used"] == 0.5
        assert "model_name" in result
        assert result["model_name"] == mock_metadata["model_name"]
        assert "timestamp" in result
        assert "status" in result
        assert result["status"] == "success"

        # Check reasonable probability (should be between 0 and 1)
        assert 0 <= result["fraud_probability"] <= 1


def test_predict_fraud_transaction():
    """Test prediction endpoint with fraud-like transaction."""
    test_transaction = generate_test_transaction(fraud_like=True)

    # Set up the mock data and model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.2, 0.8]]
    )  # 80% fraud probability

    # Mock metadata values
    mock_metadata = {
        "threshold": 0.5,
        "model_name": "Test XGBoost Model",
        "top_features": ["v14", "v4", "v12", "v10", "v11"],
        "metrics": {"accuracy": 0.95, "f1": 0.85},
    }

    # Create patches for all the necessary components
    with patch("src.api.model", mock_model), patch(
        "src.api.threshold", mock_metadata["threshold"]
    ), patch("src.api.model_name", mock_metadata["model_name"]), patch(
        "src.api.top_features", mock_metadata["top_features"]
    ), patch(
        "src.api.metadata", mock_metadata
    ), patch(
        "src.api.feature_names", [f"v{i}" for i in range(1, 29)] + ["time", "amount"]
    ):

        # Create a new TestClient for the patched app
        client = TestClient(app)  # pylint: disable=redefined-outer-name

        # Make the request
        response = client.post("/predict", json=test_transaction)

        # Check the response
        assert response.status_code == 200, f"API error: {response.text}"
        result = response.json()

        # Verify the response contains all expected fields
        assert "is_fraud" in result
        assert result["is_fraud"] is True  # 0.8 > 0.5 threshold
        assert "fraud_probability" in result
        assert result["fraud_probability"] == 0.8
        assert "threshold_used" in result
        assert result["threshold_used"] == 0.5
        assert "model_name" in result
        assert "timestamp" in result
        assert "status" in result
        assert result["status"] == "success"


def test_predict_with_custom_threshold():
    """Test prediction with a custom threshold."""
    test_transaction = generate_test_transaction()
    custom_threshold = 0.1  # Low threshold should increase chance of fraud detection

    # Set up the mock data and model
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.85, 0.15]]
    )  # 15% probability of fraud

    # Mock metadata values
    mock_metadata = {
        "threshold": 0.5,  # Default threshold, will be overridden by custom_threshold
        "model_name": "Test XGBoost Model",
        "top_features": ["v14", "v4", "v12", "v10", "v11"],
        "metrics": {"accuracy": 0.95, "f1": 0.85},
    }

    # Create patches for all the necessary components
    with patch("src.api.model", mock_model), patch(
        "src.api.threshold", mock_metadata["threshold"]
    ), patch("src.api.model_name", mock_metadata["model_name"]), patch(
        "src.api.top_features", mock_metadata["top_features"]
    ), patch(
        "src.api.metadata", mock_metadata
    ), patch(
        "src.api.feature_names", [f"v{i}" for i in range(1, 29)] + ["time", "amount"]
    ):

        # Create a new TestClient for the patched app
        client = TestClient(app)  # pylint: disable=redefined-outer-name

        # Make the request with custom threshold
        response = client.post(
            f"/predict?custom_threshold={custom_threshold}", json=test_transaction
        )

        # Check the response
        assert response.status_code == 200, f"API error: {response.text}"
        result = response.json()

        # Verify the custom threshold was used
        assert result["threshold_used"] == custom_threshold

        # Check fraud classification is correct based on threshold
        assert result["fraud_probability"] == 0.15
        assert result["is_fraud"] is True  # 0.15 > 0.1 threshold

        # Verify other fields are present
        assert "model_name" in result
        assert "timestamp" in result
        assert "status" in result
        assert result["status"] == "success"


def test_validation_amount():
    """Test input validation for amount."""
    test_transaction = generate_test_transaction()
    test_transaction["amount"] = -100  # Invalid amount

    response = client.post("/predict", json=test_transaction)

    assert response.status_code == 422  # Validation error
    assert "amount" in response.text.lower()


def test_validation_time():
    """Test input validation for time."""
    test_transaction = generate_test_transaction()
    test_transaction["time"] = 100000  # Invalid time (> 86400)

    response = client.post("/predict", json=test_transaction)

    assert response.status_code == 422  # Validation error
    assert "time" in response.text.lower()
