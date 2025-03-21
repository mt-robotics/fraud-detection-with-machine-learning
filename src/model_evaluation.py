"""Module for evaluating fraud detection models and visualizing results."""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional  # , Tuple
import time
import shap

# import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    average_precision_score,
)
from loguru import logger

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# pylint: disable=wrong-import-position
from config.config import MODELS_DIR


class ModelEvaluator:
    """Class for evaluating and visualizing model performance."""

    def __init__(self, output_dir: Optional[Path] = None, save_plots: bool = True):
        """Initialize the model evaluator.

        Args:
            output_dir: Directory to save plots to
            save_plots: Whether to save plots to disk
        """
        self.output_dir = output_dir or MODELS_DIR
        self.save_plots = save_plots
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,  # pylint: disable=invalid_name
        y_test: np.ndarray,
        model_name: str,
        threshold: float = 0.5,
        feature_names: Optional[List[str]] = None,
        business_cost_ratio: float = 5.0,  # Cost ratio of false negatives to false positives
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation with metrics and visualizations.

        Args:
            model: Trained model with predict_proba method
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for display and filenames
            threshold: Classification threshold
            feature_names: List of feature names for SHAP analysis
            business_cost_ratio: Ratio of FN cost to FP cost (e.g., 5.0 means a false negative
                                 costs 5x more than a false positive)

        Returns:
            Dictionary of evaluation metrics and plots
        """
        # Get predictions
        start_time = time.time()
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        predict_time = time.time() - start_time

        # Calculate standard metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate custom business cost metric
        # Higher cost for false negatives (missed fraud)
        business_cost = fp + (business_cost_ratio * fn)

        # Normalize to a 0-1 scale where lower is better (invert for consistency with other metrics)
        # Assume worst case is all errors (all actual fraud missed + all non-fraud flagged)
        worst_case_cost = len(y_test[y_test == 0]) + (
            business_cost_ratio * len(y_test[y_test == 1])
        )
        business_metric = 1.0 - (business_cost / worst_case_cost)

        # Store metrics
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
            "avg_precision": avg_precision,
            "business_metric": business_metric,
            "predict_time": predict_time,
            "threshold": threshold,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        }

        # Generate plots
        if self.save_plots:
            logger.info(f"Generating evaluation plots for {model_name}")
            self.plot_confusion_matrix(y_test, y_pred, model_name)
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)

            # Generate SHAP plots if feature names are provided
            if feature_names is not None:
                try:
                    self.plot_shap_summary(
                        model, X_test[:500], feature_names, model_name
                    )
                except Exception as e:
                    logger.warning(f"SHAP analysis failed: {str(e)}")

        # Log results
        logger.info(f"Model: {model_name} (threshold={threshold:.4f})")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        logger.info(f"Average Precision: {avg_precision:.4f}")
        logger.info(f"Business Metric (cost-weighted): {business_metric:.4f}")
        logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        logger.info(f"Prediction Time: {predict_time:.4f} seconds")
        logger.info(
            f"\nClassification Report:\n{classification_report(y_test, y_pred)}"
        )

        return metrics

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str
    ) -> None:
        """Plot confusion matrix for model predictions.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            model_name: Name of the model being evaluated.
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        if self.save_plots:
            plt.savefig(
                self.output_dir / f"{model_name}_confusion_matrix.png",
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def plot_roc_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str
    ) -> float:
        """Plot ROC curve for a model.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            model_name: Name of the model.

        Returns:
            AUC-ROC score
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if self.save_plots:
            plt.savefig(
                self.output_dir / f"{model_name}_roc_curve.png", bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

        return roc_auc

    def plot_precision_recall_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str
    ) -> float:
        """Plot Precision-Recall curve for a model.

        Args:
            y_true: True labels.
            y_pred_proba: Predicted probabilities.
            model_name: Name of the model.

        Returns:
            Average precision score
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        # Calculate F1 scores for each threshold
        f1_scores = np.zeros_like(precision[:-1])
        for i in range(len(thresholds)):
            f1_scores[i] = (
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i] + 1e-8)
            )

        # Find optimal threshold for F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        max_f1 = f1_scores[optimal_idx]

        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
        plt.scatter(
            recall[optimal_idx],
            precision[optimal_idx],
            color="red",
            marker="o",
            label=f"Best F1 = {max_f1:.4f} (threshold = {optimal_threshold:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {model_name}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

        if self.save_plots:
            plt.savefig(
                self.output_dir / f"{model_name}_precision_recall_curve.png",
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        logger.info(
            f"Optimal threshold for {model_name} based on PR curve: {optimal_threshold:.4f} (F1 = {max_f1:.4f})"
        )
        return avg_precision

    def plot_roc_curves(
        self,
        models_dict: Dict,
        X_val: np.ndarray,  # pylint: disable=invalid-name
        y_val: np.ndarray,
    ) -> None:
        """Plot ROC curves for multiple models.

        Args:
            models_dict: Dictionary of trained models.
            X_val: Validation features.
            y_val: Validation labels.
        """
        plt.figure(figsize=(10, 8))

        for model_name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        if self.save_plots:
            plt.savefig(
                self.output_dir / "model_comparison_roc_curves.png", bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

    def plot_feature_importance(
        self, model, feature_names: List[str], model_name: str
    ) -> None:
        """Plot feature importance for tree-based models.

        Args:
            model: Trained model with feature_importances_ attribute.
            feature_names: List of feature names.
            model_name: Name of the model.
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning(f"{model_name} does not support feature importance")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Show top 20 features

        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance - {model_name}")
        plt.barh(range(len(indices)), importances[indices], color="#2166ac")
        plt.yticks(
            range(len(indices)),
            [feature_names[i] for i in indices],
        )
        plt.xlabel("Importance")
        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                self.output_dir / f"{model_name}_feature_importance.png",
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

    def plot_shap_summary(
        self, model, X_sample: np.ndarray, feature_names: List[str], model_name: str
    ) -> None:
        """Generate SHAP summary plots to explain model predictions.

        Args:
            model: Trained model
            X_sample: Sample of data for SHAP analysis
            feature_names: List of feature names
            model_name: Name of the model
        """
        logger.info(f"Generating SHAP visualizations for {model_name}")

        try:
            # Create SHAP explainer
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

            # Summary plot (dot plot)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_sample, feature_names=feature_names, show=False
            )
            plt.title(f"SHAP Feature Importance - {model_name}")

            if self.save_plots:
                plt.savefig(
                    self.output_dir / f"{model_name}_shap_summary.png",
                    bbox_inches="tight",
                )
                plt.close()
            else:
                plt.show()

            # Bar plot of mean absolute SHAP values
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            plt.title(f"Mean Impact on Model Output Magnitude - {model_name}")

            if self.save_plots:
                plt.savefig(
                    self.output_dir / f"{model_name}_shap_bar.png", bbox_inches="tight"
                )
                plt.close()
            else:
                plt.show()

            # Get top 5 features by mean absolute SHAP value
            # mean_abs_shap = np.abs(shap_values.values).mean(0)
            mean_abs_shap = np.abs(shap_values).mean(0)
            top_features_idx = np.argsort(mean_abs_shap)[-5:][
                ::-1
            ]  # Get indices of top 5 features
            top_features = []
            for i in top_features_idx:
                top_features.append(feature_names[int(i)])  # Convert index to integer

            logger.info(f"Top 5 features by SHAP importance: {', '.join(top_features)}")

            # Save top features
            top_features_path = self.output_dir / f"{model_name}_top_features.txt"
            with open(top_features_path, "w", encoding="utf-8") as f:
                for i, feature in enumerate(top_features):
                    f.write(
                        f"{i+1}. {feature} - {mean_abs_shap[top_features_idx[i]]:.4f}\n"
                    )

            # Plot dependence plots for top features
            for i, feature in enumerate(top_features):
                feature_idx = feature_names.index(feature)
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    feature_idx,
                    shap_values,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                )
                plt.title(f"SHAP Dependence Plot for {feature}")

                if self.save_plots:
                    plt.savefig(
                        self.output_dir / f"{model_name}_shap_dependence_{feature}.png",
                        bbox_inches="tight",
                    )
                    plt.close()
                else:
                    plt.show()

        except Exception as e:
            logger.error(f"Error in SHAP analysis: {str(e)}")
            raise

    def compare_models_metrics(
        self, results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """Create a comparison plot of different metrics across models.

        Args:
            results: Dictionary containing evaluation metrics for all models.

        Returns:
            DataFrame with metrics for all models
        """
        # Convert to DataFrame for easier manipulation
        comparison_df = pd.DataFrame(results).T

        # Plot metrics comparison
        metrics_to_plot = ["precision", "recall", "f1", "auc_roc"]
        titles = ["Precision", "Recall", "F1 Score", "AUC-ROC"]

        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
            ax = axes[i]
            comparison_df[metric].plot(
                kind="bar", ax=ax, color=sns.color_palette("Set2")
            )
            ax.set_title(title)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for j, v in enumerate(comparison_df[metric]):
                ax.text(j, v + 0.02, f"{v:.3f}", ha="center")

        plt.tight_layout()

        if self.save_plots:
            plt.savefig(
                self.output_dir / "model_comparison_metrics.png", bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

        # Print metrics table
        logger.info("\nModel Comparison:")
        logger.info(
            comparison_df[
                ["precision", "recall", "f1", "auc_roc", "predict_time"]
            ].to_string()
        )

        return comparison_df

    def find_optimal_threshold(
        self,
        model,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_name: str,
        metric: str = "f2",
    ) -> Dict[str, float]:
        """Find the optimal threshold for classification.

        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            model_name: Name of the model
            metric: Metric to optimize ('f1', 'f2', or 'precision_recall_sum')

        Returns:
            Dictionary with optimal thresholds for different metrics
        """
        logger.info(f"Finding optimal threshold for {model_name}")

        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        thresholds = np.append(
            thresholds, 1.0
        )  # Add 1.0 to match precision/recall arrays

        # Calculate F1 score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Calculate F2 score (weights recall higher)
        beta = 2
        f2_scores = (
            (1 + beta**2)
            * (precision * recall)
            / ((beta**2 * precision) + recall + 1e-8)
        )

        # Find optimal thresholds
        max_f1_idx = np.argmax(f1_scores)
        max_f2_idx = np.argmax(f2_scores)
        max_pr_sum_idx = np.argmax(precision + recall)

        # Get optimal thresholds
        optimal_thresholds = {
            "f1": thresholds[max_f1_idx],
            "f2": thresholds[max_f2_idx],
            "precision_recall_sum": thresholds[max_pr_sum_idx],
            "default": 0.5,
        }

        # Plot thresholds vs metrics
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds[:-1], precision[:-1], "b-", label="Precision")
        plt.plot(thresholds[:-1], recall[:-1], "g-", label="Recall")
        plt.plot(thresholds[:-1], f1_scores[:-1], "r-", label="F1 Score")
        plt.plot(thresholds[:-1], f2_scores[:-1], "y-", label="F2 Score")

        # Plot vertical lines for optimal thresholds
        plt.axvline(x=optimal_thresholds["f1"], linestyle="--", color="red", alpha=0.5)
        plt.axvline(
            x=optimal_thresholds["f2"], linestyle="--", color="yellow", alpha=0.5
        )
        plt.axvline(
            x=optimal_thresholds["precision_recall_sum"],
            linestyle="--",
            color="green",
            alpha=0.5,
        )
        plt.axvline(x=0.5, linestyle="--", color="black", alpha=0.5)

        # Add labels
        plt.text(
            optimal_thresholds["f1"],
            0.2,
            f"F1: {optimal_thresholds['f1']:.2f}",
            rotation=90,
            alpha=0.7,
        )
        plt.text(
            optimal_thresholds["f2"],
            0.2,
            f"F2: {optimal_thresholds['f2']:.2f}",
            rotation=90,
            alpha=0.7,
        )
        plt.text(
            optimal_thresholds["precision_recall_sum"],
            0.2,
            f"PR Sum: {optimal_thresholds['precision_recall_sum']:.2f}",
            rotation=90,
            alpha=0.7,
        )
        plt.text(0.5, 0.2, "Default: 0.5", rotation=90, alpha=0.7)

        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Metrics vs Threshold - {model_name}")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)

        if self.save_plots:
            plt.savefig(
                self.output_dir / f"{model_name}_threshold_optimization.png",
                bbox_inches="tight",
            )
            plt.close()
        else:
            plt.show()

        # Recommended threshold based on metric
        recommended = optimal_thresholds[metric]
        logger.info(f"Recommended threshold ({metric}): {recommended:.4f}")

        # Log all thresholds
        for name, threshold in optimal_thresholds.items():
            logger.info(f"{name} optimal threshold: {threshold:.4f}")

        return optimal_thresholds
