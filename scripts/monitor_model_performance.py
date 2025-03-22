#!/usr/bin/env python
"""
Model Performance Monitoring Script

This script monitors the performance of the fraud detection model in production.
It compares recent predictions with ground truth data to detect model drift.
"""

import argparse
import logging

# import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/model_monitoring.log"),
    ],
)
logger = logging.getLogger(__name__)

# Add root directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import MODELS_DIR  # pylint: disable=wrong-import-position


def load_model_metadata():
    """Load model metadata from file."""
    metadata_path = MODELS_DIR / "model_metadata.joblib"
    if not metadata_path.exists():
        logger.error("Model metadata file not found: %s", metadata_path)
        return None

    try:
        metadata = joblib.load(metadata_path)
        logger.info(
            "Loaded model metadata for %s", {metadata.get("model_name", "unknown")}
        )
        return metadata
    except Exception as e:
        logger.error("Failed to load model metadata: %s", str(e))
        return None


def load_production_data(days_back=1, data_path=None):
    """
    Load production data with ground truth labels for monitoring.

    Args:
        days_back: How many days of data to analyze
        data_path: Optional path to data file, if not using default location

    Returns:
        DataFrame with production data
    """
    # In a real system, this would load from a database or data lake
    # For demonstration, we'll load from a CSV file

    if data_path:
        file_path = Path(data_path)
    else:
        # Default to a production data location
        file_path = Path("data/production/transactions.csv")

    if not file_path.exists():
        logger.error("Production data file not found: %s", file_path)
        # Create dummy data for demonstration
        logger.info("Creating synthetic data for demonstration")

        # Create synthetic data with 5% fraud rate
        np.random.seed(42)
        n_samples = 1000

        # Features
        X = np.random.randn(n_samples, 30)  # pylint: disable=invalid-name

        # Target with 5% fraud
        y = np.zeros(n_samples)
        fraud_indices = np.random.choice(
            range(n_samples), size=int(0.05 * n_samples), replace=False
        )
        y[fraud_indices] = 1

        # Create DataFrame with proper column names
        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        df = pd.DataFrame(np.column_stack([X, y]), columns=columns)

        # Add prediction and timestamp columns for monitoring
        df["predicted"] = np.random.choice([0, 1], size=n_samples, p=[0.94, 0.06])
        df["prediction_time"] = pd.Timestamp.now() - timedelta(days=days_back / 2)

        return df

    try:
        df = pd.read_csv(file_path)

        # Filter for recent data based on days_back
        if "prediction_time" in df.columns:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df["prediction_time"] = pd.to_datetime(df["prediction_time"])
            df = df[df["prediction_time"] >= cutoff_date]

        logger.info("Loaded %d production data records for analysis", len(df))
        return df
    except Exception as e:
        logger.error("Failed to load production data: %s", str(e))
        return None


def calculate_metrics(y_true, y_pred, y_score=None):
    """Calculate model performance metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["auc_roc"] = 0.5  # Default for failed AUC calculation

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    return metrics


def check_for_drift(baseline_metrics, current_metrics, threshold=0.05):
    """
    Check for model drift by comparing baseline and current metrics.

    Args:
        baseline_metrics: Metrics from model training
        current_metrics: Metrics from production data
        threshold: Threshold for acceptable metric difference

    Returns:
        Dictionary of drift findings
    """
    drift_results = {"has_drift": False, "metrics_diff": {}}

    # Compare key metrics
    for metric in ["f1", "precision", "recall", "auc_roc"]:
        if metric in baseline_metrics and metric in current_metrics:
            diff = abs(baseline_metrics[metric] - current_metrics[metric])
            drift_results["metrics_diff"][metric] = diff

            if diff > threshold:
                drift_results["has_drift"] = True

    return drift_results


def generate_monitoring_report(baseline_metrics, current_metrics, drift_results):
    """Generate a detailed monitoring report."""
    report = []
    report.append("=" * 50)
    report.append("FRAUD DETECTION MODEL PERFORMANCE MONITORING")
    report.append("=" * 50)
    report.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("BASELINE METRICS (FROM MODEL TRAINING)")
    report.append("-" * 40)
    for metric, value in baseline_metrics.items():
        if isinstance(value, (int, float)):
            report.append(f"{metric}: {value:.4f}")
    report.append("")

    report.append("CURRENT PRODUCTION METRICS")
    report.append("-" * 40)
    for metric, value in current_metrics.items():
        if isinstance(value, (int, float)):
            report.append(f"{metric}: {value:.4f}")
    report.append("")

    report.append("DRIFT ANALYSIS")
    report.append("-" * 40)
    report.append(f"Drift Detected: {'YES' if drift_results['has_drift'] else 'NO'}")
    report.append("Metric Differences:")
    for metric, diff in drift_results.get("metrics_diff", {}).items():
        status = "ALERT" if diff > 0.05 else "OK"
        report.append(f"  {metric}: {diff:.4f} [{status}]")

    return "\n".join(report)


def main(args):  # pylint: disable=redefined-outer-name
    """Main function for model monitoring."""
    logger.info("Starting model performance monitoring")

    # Load model metadata with baseline metrics
    metadata = load_model_metadata()
    if not metadata or "metrics" not in metadata:
        logger.error("Cannot proceed without model metadata and baseline metrics")
        return 1

    baseline_metrics = metadata.get("metrics", {})
    logger.info("Baseline F1 score: %s", baseline_metrics.get("f1", "unknown"))

    # Load production data
    production_data = load_production_data(
        days_back=args.days, data_path=args.data_path
    )
    if production_data is None or len(production_data) == 0:
        logger.error("No production data available for analysis")
        return 1

    # Extract ground truth and predictions
    if (
        "Class" not in production_data.columns
        or "predicted" not in production_data.columns
    ):
        logger.error(
            "Production data missing required columns: 'Class' and 'predicted'"
        )
        return 1

    y_true = production_data["Class"]
    y_pred = production_data["predicted"]

    # Calculate current metrics
    current_metrics = calculate_metrics(y_true, y_pred)
    logger.info("Current F1 score: %s", current_metrics.get("f1", "unknown"))

    # Check for drift
    drift_results = check_for_drift(
        baseline_metrics, current_metrics, threshold=args.threshold
    )

    # Generate and save report
    report = generate_monitoring_report(
        baseline_metrics, current_metrics, drift_results
    )

    # Print report
    print(report)

    # Save report to file
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = (
        report_dir / f"model_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Monitoring report saved to %s", report_path)

    # Return error code if drift detected
    if args.alert_on_drift and drift_results["has_drift"]:
        logger.warning("Model drift detected!")
        return 2

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitor fraud detection model performance"
    )
    parser.add_argument(
        "--days", type=int, default=1, help="Number of days of data to analyze"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.05, help="Threshold for drift detection"
    )
    parser.add_argument("--data-path", type=str, help="Path to production data file")
    parser.add_argument(
        "--alert-on-drift",
        action="store_true",
        help="Return error code if drift detected",
    )

    args = parser.parse_args()
    sys.exit(main(args))
