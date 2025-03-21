import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# pylint: disable=wrong-import-position
from src.data_preprocessing import (
    preprocess_data,
)

# from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame(
        {
            "time": np.random.rand(100),
            "v1": np.random.rand(100),
            "amount": np.random.rand(100),
            "Class": np.random.binomial(1, 0.1, 100),
        }
    )


def test_preprocess_data(sample_data):  # pylint: disable=redefined-outer-name
    """Test data preprocessing function."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test, scaler = preprocess_data(sample_data)

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # With SMOTE, the training set size will be different from the original
    # The number of samples in the balanced dataset should be greater than
    # or equal to the original training set size
    original_train_size = int(len(sample_data) * 0.8)  # Default test_size is 0.2
    assert X_train.shape[0] >= original_train_size

    # Test set size should be 20% of the original data
    assert X_test.shape[0] == int(len(sample_data) * 0.2)

    # Check that the scaler was properly created
    assert hasattr(scaler, "transform")
    assert hasattr(scaler, "fit_transform")
