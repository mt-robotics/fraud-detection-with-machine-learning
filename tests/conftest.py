"""Shared setup and mock objects for fraud detection tests, like a backstage crew preparing the stage for a smooth performance."""  # pylint: disable=line-too-long

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch  # patch model before importing the app
import pytest
import numpy as np

# import joblib

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


@pytest.fixture
def mock_model():
    """Create a mock model for API testing."""
    # Create a mock model that returns predictable values
    mock = MagicMock()

    # Return values should be NumPy arrays as real models would do
    # but our API code will handle the conversion
    mock.predict_proba.return_value = np.array([[0.9, 0.1]])  # 10% probability of fraud
    mock.predict.return_value = np.array([0])  # Predict non-fraud

    return mock


@pytest.fixture
def app_with_mock_model(mock_model):  # pylint: disable=redefined-outer-name
    """Create a FastAPI app with mocked model for testing."""

    # Set up mock data and configuration values
    mock_metadata = {
        "threshold": 0.5,
        "model_name": "XGBoost Test Model",
        "top_features": ["v14", "v4", "v12", "v10", "v11"],
        "metrics": {"accuracy": 0.95, "f1": 0.85, "precision": 0.90, "recall": 0.80},
        "training_date": "2025-03-21",
    }

    # Create a patch for the model object
    with patch.dict("sys.modules", {"src.api": None}):
        # Remove the api module from sys.modules if it's there
        sys.modules.pop("src.api", None)

        # Mock the important variables
        # Set up multiple patches to ensure all the necessary objects are properly mocked
        with patch(
            "joblib.load",
            side_effect=lambda path: (
                mock_model if "best_model" in str(path) else mock_metadata
            ),
        ), patch("src.api.model", mock_model), patch(
            "src.api.threshold", mock_metadata["threshold"]
        ), patch(
            "src.api.model_name", mock_metadata["model_name"]
        ), patch(
            "src.api.top_features", mock_metadata["top_features"]
        ), patch(
            "src.api.feature_names",
            [f"v{i}" for i in range(1, 29)] + ["time", "amount"],
        ), patch(
            "src.api.metadata", mock_metadata
        ):

            # Now import the app with patches in place
            from src.api import app  # pylint: disable=import-outside-toplevel

            yield app
