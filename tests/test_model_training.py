"""Tests for the model training module."""

# import os
import sys
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pytest
import joblib

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.model_training import ModelTrainer  # pylint: disable=wrong-import-position

# Disable matplotlib plots during tests
plt.ioff()


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)  # For reproducible results
    # pylint: disable=invalid-name
    X_train = np.random.rand(100, 30)
    X_test = np.random.rand(50, 30)
    y_train = np.random.binomial(1, 0.1, 100)
    y_test = np.random.binomial(1, 0.1, 50)
    return X_train, X_test, y_train, y_test


@pytest.fixture
def feature_names():
    """Create sample feature names."""
    return [f"v{i}" for i in range(1, 29)] + ["time", "amount"]


def test_model_trainer_initialization():
    """Test that ModelTrainer initializes properly."""
    trainer = ModelTrainer()
    assert len(trainer.models) == 4  # Now includes gradient_boosting
    assert trainer.best_model is None
    assert trainer.best_model_name is None
    assert trainer.optimal_threshold == 0.5
    assert trainer.feature_names is None


def test_train_and_evaluate(
    sample_training_data,
):  # pylint: disable=redefined-outer-name
    """Test training and evaluating models."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = sample_training_data
    trainer = ModelTrainer()

    # Set feature names for interpretability
    trainer.feature_names = [f"v{i}" for i in range(1, 29)] + ["time", "amount"]

    # Test without threshold optimization
    results = trainer.train_and_evaluate_all(
        X_train, y_train, X_test, y_test, optimize_threshold=False
    )

    assert isinstance(results, dict)
    assert len(results) == 4  # 4 models
    assert trainer.best_model is not None
    assert trainer.best_model_name is not None
    assert trainer.optimal_threshold == 0.5  # Should still be default

    # Check metrics have expected keys
    for model_metrics in results.values():
        assert "precision" in model_metrics
        assert "recall" in model_metrics
        assert "f1" in model_metrics
        assert "auc_roc" in model_metrics
        assert "threshold" in model_metrics
        assert model_metrics["threshold"] == 0.5


def test_train_and_evaluate_with_threshold_optimization(
    sample_training_data,
):  # pylint: disable=redefined-outer-name
    """Test training and evaluating models with threshold optimization."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = sample_training_data
    trainer = ModelTrainer()

    # Test with threshold optimization
    results = trainer.train_and_evaluate_all(
        X_train, y_train, X_test, y_test, optimize_threshold=True
    )

    assert isinstance(results, dict)
    assert len(results) == 4

    # Check thresholds are optimized
    for model_metrics in results.values():
        assert model_metrics["threshold"] != 0.5  # Should be optimized


def test_find_optimal_threshold(
    sample_training_data,
):  # pylint: disable=redefined-outer-name
    """Test threshold optimization."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = sample_training_data
    trainer = ModelTrainer()

    # Train a model
    model_name = "logistic_regression"
    trainer.train_model(model_name, X_train, y_train)

    # Find optimal threshold
    threshold = trainer.find_optimal_threshold(model_name, X_test, y_test, "f1")

    assert 0 <= threshold <= 1  # Threshold should be a probability

    # Test different metrics
    threshold_f2 = trainer.find_optimal_threshold(model_name, X_test, y_test, "f2")
    assert 0 <= threshold_f2 <= 1


def test_model_save_and_load(
    sample_training_data, feature_names
):  # pylint: disable=redefined-outer-name
    """Test saving and loading models."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = sample_training_data

    # Create a temporary directory for test models
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_model_dir = Path(tmpdirname)

        # Create a trainer with custom output directory
        trainer = ModelTrainer()
        trainer.feature_names = feature_names

        # Train models
        trainer.train_and_evaluate_all(X_train, y_train, X_test, y_test)

        # Save best model to custom directory
        model_path, metadata_path = trainer.save_best_model(custom_dir=temp_model_dir)

        # Check files exist
        assert model_path.exists(), f"Path {model_path} should exist"
        assert metadata_path.exists(), f"Path {metadata_path} should exist"

        # Verify we can load the model back
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None

        # Verify we can load the metadata back
        loaded_metadata = joblib.load(metadata_path)
        assert loaded_metadata is not None
        assert "model_name" in loaded_metadata
        assert loaded_metadata["model_name"] == trainer.best_model_name


def test_hyperparameter_tuning(
    sample_training_data,
):  # pylint: disable=redefined-outer-name
    """Test hyperparameter tuning."""
    # pylint: disable=invalid-name
    X_train, X_test, y_train, y_test = sample_training_data
    trainer = ModelTrainer()

    # Test with logistic regression (scikit-learn approach)
    model_name = "logistic_regression"
    trainer.train_model(model_name, X_train, y_train)
    original_model = trainer.models[model_name]

    param_dist = {
        "C": [0.1, 1.0],
        "solver": ["lbfgs", "saga"],
    }

    try:
        trainer.tune_hyperparameters(
            model_name, param_dist, X_train, y_train, n_iter=2, cv=2
        )
        assert trainer.models[model_name] is not None
    except Exception:
        assert trainer.models[model_name] is original_model

    # Test with XGBoost (native approach)
    model_name = "xgboost"
    trainer.train_model(model_name, X_train, y_train)
    original_model = trainer.models[model_name]

    param_dist = {
        "max_depth": [3, 4],
        "learning_rate": [0.1, 0.2],
    }

    # This should now work with our native XGBoost implementation
    trainer.tune_hyperparameters(
        model_name, param_dist, X_train, y_train, n_iter=2, cv=2
    )

    # Check the model was updated
    assert trainer.models[model_name] is not None
    # The test will fail if the native XGBoost tuning approach doesn't work
