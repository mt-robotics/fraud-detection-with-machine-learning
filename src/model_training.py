# pylint: disable=too-many-lines
"""Module for model training and evaluation with advanced methods."""

from typing import Dict, List, Optional, Tuple
import time
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    fbeta_score,
    # make_scorer,
)
import xgboost as xgb
import lightgbm as lgb
import joblib
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# pylint: disable=wrong-import-position
from config.config import (
    MODELS_DIR,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    BUSINESS_COST_RATIO,
    PRIMARY_METRIC,
)


# Define a custom F2 score for use as a scorer
def f2_score(y_true, y_pred):
    """Calculate F2 score - weighted F-score with beta=2."""
    return fbeta_score(y_true, y_pred, beta=2)


# Create a custom business metric scorer
def business_cost_score(y_true, y_pred, business_cost_ratio=5.0):
    """Calculate a custom business score based on relative costs of FP and FN.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        business_cost_ratio: Cost of FN relative to FP (default: 5.0)

    Returns:
        Score between 0 and 1, higher is better
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate business cost
    business_cost = fp + (business_cost_ratio * fn)

    # Normalize to a 0-1 scale where higher is better
    worst_case_cost = len(y_true[y_true == 0]) + (
        business_cost_ratio * len(y_true[y_true == 1])
    )
    business_score = 1.0 - (business_cost / worst_case_cost)

    return business_score


class ModelTrainer:
    """Class for training and evaluating fraud detection models with ensemble methods."""

    def __init__(
        self, custom_models: Optional[Dict] = None, advanced_models: bool = True
    ):
        """Initialize the model trainer with different model types.

        Args:
            custom_models: Optional dictionary of custom models to use
            advanced_models: Whether to include advanced and ensemble models
        """
        # Define base models
        base_models = {
            "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            "random_forest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            "xgboost": xgb.XGBClassifier(**XGBOOST_PARAMS),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
        }

        # If advanced models are requested, add more sophisticated models
        if advanced_models:
            # Neural network classifier
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation="relu",
                solver="adam",
                alpha=0.0001,
                max_iter=200,
                random_state=RANDOM_STATE,
            )

            # Calibrated SVC - better probability estimates
            calibrated_svc = CalibratedClassifierCV(
                SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
                method="sigmoid",
                cv=3,
            )

            # Voting ensemble (soft voting)
            voting_clf = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=RANDOM_STATE
                        ),
                    ),
                    (
                        "xgb",
                        xgb.XGBClassifier(
                            use_label_encoder=False,
                            eval_metric="logloss",
                            n_estimators=100,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ],
                voting="soft",  # Use probability estimates
                n_jobs=-1,
            )

            # Stacking ensemble with meta-learner
            stacking_clf = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=RANDOM_STATE
                        ),
                    ),
                    (
                        "xgb",
                        xgb.XGBClassifier(
                            use_label_encoder=False,
                            eval_metric="logloss",
                            n_estimators=100,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ],
                final_estimator=LogisticRegression(C=10.0),
                cv=3,
                n_jobs=-1,
            )

            # Add these advanced models to the base models
            advanced_model_dict = {
                "neural_network": mlp,
                "calibrated_svc": calibrated_svc,
                "voting_ensemble": voting_clf,
                "stacking_ensemble": stacking_clf,
            }

            # Merge the dictionaries
            base_models.update(advanced_model_dict)

        # Use custom models if provided, otherwise use the built-in ones
        self.models = custom_models or base_models

        # State tracking
        self.best_model = None
        self.best_model_name = None
        self.best_score = -np.inf
        self.optimal_threshold = 0.5
        self.feature_names = None
        self.metrics_dict = {}
        self.shap_values = None
        self.preprocessor = None

    def train_model(
        self,
        # pylint: disable=redefined-outer-name
        model_name: str,
        X_train: np.ndarray,  # pylint: disable=invalid-name
        y_train: np.ndarray,
    ) -> None:
        """Train a specific model.

        Args:
            model_name: Name of the model to train.
            X_train: Training features.
            y_train: Training labels.
        """
        logger.info(f"Training {model_name} model")
        model = self.models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(
            f"{model_name} model training completed in {train_time:.2f} seconds"
        )

    def evaluate_model(
        self,
        model_name: str,  # pylint: disable=redefined-outer-name
        X_val: np.ndarray,  # pylint: disable=invalid-name
        y_val: np.ndarray,
        threshold: float = 0.5,
        business_cost_ratio: float = 5.0,  # Cost ratio of false negatives to false positives
    ) -> Dict[str, float]:
        """Evaluate a trained model.

        Args:
            model_name: Name of the model to evaluate.
            X_val: Validation features.
            y_val: Validation labels.
            threshold: Classification threshold (default: 0.5)
            business_cost_ratio: Ratio of FN cost to FP cost (higher value means FN is more costly)

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info(f"Evaluating {model_name} model with threshold {threshold}")
        model = self.models[model_name]

        # Get predictions
        start_time = time.time()
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        predict_time = time.time() - start_time

        # Calculate standard metrics
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        avg_precision = average_precision_score(y_val, y_pred_proba)

        # Calculate confusion matrix for business cost metric
        from sklearn.metrics import confusion_matrix

        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        # Calculate custom business cost metric
        # Higher cost for false negatives (missed fraud)
        business_cost = fp + (business_cost_ratio * fn)

        # Normalize to a 0-1 scale where higher is better
        worst_case_cost = len(y_val[y_val == 0]) + (
            business_cost_ratio * len(y_val[y_val == 1])
        )
        business_metric = 1.0 - (business_cost / worst_case_cost)

        # Compile metrics
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

        logger.info(f"{model_name} evaluation metrics:\n{metrics}")
        return metrics

    def find_optimal_threshold(
        self, model_name: str, X_val: np.ndarray, y_val: np.ndarray, metric: str = "f2"
    ) -> float:
        """Find the optimal threshold for classification.

        Args:
            model_name: Name of the model to optimize.
            X_val: Validation features.
            y_val: Validation labels.
            metric: Metric to optimize ('f1', 'f2', or 'precision_recall_sum').

        Returns:
            Optimal threshold value
        """
        logger.info(f"Finding optimal threshold for {model_name} based on {metric}")
        model = self.models[model_name]
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Calculate precision and recall at different thresholds
        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_val, y_pred_proba
        )
        thresholds = np.append(
            thresholds, 1.0
        )  # Add 1.0 to match precision/recall arrays

        # F1 score
        f1_scores = (
            2
            * (precision_curve * recall_curve)
            / (precision_curve + recall_curve + 1e-8)
        )

        # F2 score (weights recall higher)
        beta = 2  # F2 gives recall higher weight
        f2_scores = (
            (1 + beta**2)
            * (precision_curve * recall_curve)
            / ((beta**2 * precision_curve) + recall_curve + 1e-8)
        )

        # Simple sum
        pr_sum = precision_curve + recall_curve

        # Find optimal threshold
        if metric == "f1":
            optimal_idx = np.argmax(f1_scores)
        elif metric == "f2":
            optimal_idx = np.argmax(f2_scores)
        else:  # precision_recall_sum
            optimal_idx = np.argmax(pr_sum)

        optimal_threshold = thresholds[optimal_idx]
        logger.info(f"Optimal threshold for {model_name} is {optimal_threshold:.4f}")

        return optimal_threshold

    def train_and_evaluate_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimize_threshold: bool = True,
        best_metric: str = "f1",
    ) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            optimize_threshold: Whether to find the optimal threshold for each model
            best_metric: Metric to use for selecting the best model ('f1', 'auc_roc', etc.)

        Returns:
            Dictionary containing evaluation metrics for all models.
        """
        self.metrics_dict = {}
        for model_name in self.models.keys():
            # Train the model
            self.train_model(model_name, X_train, y_train)

            # Find optimal threshold if requested
            threshold = 0.5
            if optimize_threshold:
                threshold = self.find_optimal_threshold(model_name, X_val, y_val, "f2")

            # Evaluate the model with business cost ratio from config
            metrics = self.evaluate_model(
                model_name,
                X_val,
                y_val,
                threshold,
                business_cost_ratio=BUSINESS_COST_RATIO,
            )
            self.metrics_dict[model_name] = metrics

            # Track best model based on the selected metric
            if metrics[best_metric] > self.best_score:
                self.best_score = metrics[best_metric]
                self.best_model = self.models[model_name]
                self.best_model_name = model_name
                self.optimal_threshold = threshold

        logger.info(
            f"Best performing model: {self.best_model_name} with {best_metric}: {self.best_score:.4f}"
        )
        return self.metrics_dict

    def tune_hyperparameters(
        self,
        model_name: str,
        param_dist: Dict[str, List],
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 10,
        cv: int = 3,
        scoring: str = "f1",  # Use F1 score instead of ROC-AUC
    ) -> None:
        """Tune model hyperparameters using RandomizedSearchCV or native XGBoost CV.

        Args:
            model_name: Name of the model to tune.
            param_dist: Dictionary of hyperparameter distributions to sample from.
            X_train: Training features.
            y_train: Training labels.
            n_iter: Number of parameter settings to sample (reduced for faster tuning).
            cv: Number of cross-validation folds.
            scoring: Scoring metric to use for hyperparameter optimization.
        """
        logger.info(
            f"Tuning hyperparameters for {model_name} using {scoring} scoring..."
        )
        model = self.models[model_name]

        # Special handling for XGBoost models to avoid sklearn compatibility issues
        if model_name in ["xgboost", "gradient_boosting"]:
            import xgboost as xgb
            from itertools import product
            import random

            logger.info(f"Using native XGBoost tuning for {model_name}")

            # Convert to DMatrix for XGBoost native API
            dtrain = xgb.DMatrix(X_train, label=y_train)

            # Create parameter combinations to try
            param_keys = list(param_dist.keys())
            param_values = list(param_dist.values())

            # Generate random combinations to sample
            all_combinations = list(product(*param_values))
            if len(all_combinations) > n_iter:
                random.seed(RANDOM_STATE)
                param_combinations = random.sample(all_combinations, n_iter)
            else:
                param_combinations = all_combinations

            best_score = -np.inf
            best_params = None
            best_model = None

            start_time = time.time()

            # Base parameters (non-tuning ones)
            base_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "verbosity": 0,
                "seed": RANDOM_STATE,
            }

            # Try each parameter combination
            for i, combo in enumerate(param_combinations):
                params = base_params.copy()

                # Add the parameter combination to try
                for k, v in zip(param_keys, combo):
                    params[k] = v

                logger.info(
                    f"Trying parameter combination {i+1}/{len(param_combinations)}: {params}"
                )

                # Use XGBoost's native cross-validation
                cv_results = xgb.cv(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=100,
                    nfold=cv,
                    stratified=True,
                    early_stopping_rounds=10,
                    metrics=["auc", "error"],
                    seed=RANDOM_STATE,
                )

                # Get the best performance
                if scoring == "f1":
                    # For f1, we need predictions to calculate it
                    # Use AUC as a proxy for now
                    best_iteration = len(cv_results)
                    iteration_score = cv_results["test-auc-mean"].iloc[-1]
                else:
                    # Directly use AUC
                    best_iteration = len(cv_results)
                    iteration_score = cv_results["test-auc-mean"].iloc[-1]

                logger.info(f"Score for combination {i+1}: {iteration_score:.4f}")

                # Check if this is the best score so far
                if iteration_score > best_score:
                    best_score = iteration_score
                    best_params = {k: v for k, v in zip(param_keys, combo)}
                    best_params["n_estimators"] = best_iteration

                    # Create and train a new model with the best parameters
                    new_params = base_params.copy()
                    new_params.update(best_params)
                    best_model = xgb.XGBClassifier(**new_params)
                    best_model.fit(X_train, y_train)

            tuning_time = time.time() - start_time

            # Log results
            logger.info(
                f"Hyperparameter tuning for {model_name} completed in {tuning_time:.2f} seconds"
            )
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV score: {best_score:.4f}")

            # Update the model
            if best_model is not None:
                self.models[model_name] = best_model

            return

        # For non-XGBoost models, use scikit-learn's RandomizedSearchCV
        try:
            # Create the RandomizedSearchCV object
            cv_obj = StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=RANDOM_STATE
            )

            # Use pre-dispatch to limit parallel jobs and avoid memory issues
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=-1,
                cv=cv_obj,
                random_state=RANDOM_STATE,
                verbose=1,
                pre_dispatch="2*n_jobs",  # Limit memory usage
                error_score="raise",
            )

            # Fit the model
            start_time = time.time()
            search.fit(X_train, y_train)
            tuning_time = time.time() - start_time

            # Log results
            logger.info(
                f"Hyperparameter tuning for {model_name} completed in {tuning_time:.2f} seconds"
            )
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score ({scoring}): {search.best_score_:.4f}")

            # Update the model
            self.models[model_name] = search.best_estimator_
        except Exception as e:
            logger.warning(
                f"Hyperparameter tuning for {model_name} failed: {str(e)}. Using current model."
            )

    def save_model(self, model_name: str) -> None:
        """Save a trained model to disk.

        Args:
            model_name: Name of the model to save.
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(self.models[model_name], model_path)
        logger.info(f"Model saved to {model_path}")

    def generate_model_explanations(self, X_test, save_dir=None) -> None:
        """Generate model explanations using SHAP.

        Args:
            X_test: Test data for generating explanations
            save_dir: Directory to save explanation plots
        """
        if self.best_model is None:
            logger.warning("No best model found. Train models first.")
            return

        if save_dir is None:
            save_dir = MODELS_DIR

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Generating SHAP explanations for {self.best_model_name}")

            # Sample a subset of the test data for explanations (for speed)
            if len(X_test) > 500:
                np.random.seed(RANDOM_STATE)
                indices = np.random.choice(len(X_test), 500, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test

            # Create SHAP explainer based on model type
            if "xgboost" in self.best_model_name or "lightgbm" in self.best_model_name:
                explainer = shap.TreeExplainer(self.best_model)
            elif "neural_network" in self.best_model_name:
                explainer = shap.DeepExplainer(self.best_model, X_sample)
            else:
                # For other models, use KernelExplainer
                # Create a prediction function that returns probabilities
                def model_predict(X):
                    return self.best_model.predict_proba(X)[:, 1]

                explainer = shap.KernelExplainer(
                    model_predict, shap.sample(X_sample, 100)
                )

            # Generate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Store for later use
            self.shap_values = shap_values

            # Create a DataFrame with feature names for better visualizations
            if self.feature_names is not None:
                X_sample_df = pd.DataFrame(X_sample, columns=self.feature_names)
            else:
                X_sample_df = pd.DataFrame(X_sample)

            # Summary plot
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # For multi-class models
                shap.summary_plot(shap_values[1], X_sample_df, show=False)
            else:
                # For binary classification
                shap.summary_plot(shap_values, X_sample_df, show=False)
            plt.tight_layout()
            plt.savefig(save_dir / "shap_summary.png")
            plt.close()

            # Dependence plots for top features
            if self.feature_names is not None:
                # Determine feature importance based on mean absolute SHAP values
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    # For multi-class, use class 1 (fraud)
                    feature_importance = np.abs(shap_values[1]).mean(0)
                else:
                    feature_importance = np.abs(shap_values).mean(0)

                # Get top 5 features by importance
                if len(feature_importance) == len(self.feature_names):
                    top_indices = np.argsort(feature_importance)[-5:]
                    top_features = [self.feature_names[i] for i in top_indices]

                    # Create dependence plots for top features
                    for feat in top_features:
                        plt.figure(figsize=(12, 8))
                        if isinstance(shap_values, list) and len(shap_values) > 1:
                            shap.dependence_plot(
                                feat, shap_values[1], X_sample_df, show=False
                            )
                        else:
                            shap.dependence_plot(
                                feat, shap_values, X_sample_df, show=False
                            )
                        plt.tight_layout()
                        plt.savefig(save_dir / f"shap_dependence_{feat}.png")
                        plt.close()

            # Store top features for metadata
            if self.feature_names is not None:
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    mean_abs_shap = np.abs(shap_values[1]).mean(0)
                else:
                    mean_abs_shap = np.abs(shap_values).mean(0)

                feature_importance = dict(zip(self.feature_names, mean_abs_shap))
                sorted_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )
                top_features = [f[0] for f in sorted_features[:10]]

                # Store top features in the object
                self.top_features = top_features

            logger.info(f"SHAP explanations saved to {save_dir}")

        except Exception as e:
            logger.warning(f"Error generating SHAP explanations: {str(e)}")
            # Continue without explanations
            self.top_features = self.feature_names[:10] if self.feature_names else []

    def save_best_model(self, custom_dir=None) -> Tuple[Path, Path]:
        """Save the best performing model to disk with metadata and explanations.

        Args:
            custom_dir: Optional custom directory to save the model to.
                       If provided, it overrides the MODELS_DIR from config.

        Returns:
            Tuple of (model_path, metadata_path)
        """
        if self.best_model is None:
            raise ValueError("No best model found. Train models first.")

        # Use the custom directory if provided, otherwise use MODELS_DIR from config
        save_dir = custom_dir if custom_dir is not None else MODELS_DIR

        # Make sure the directory exists
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        model_path = save_dir / "best_model.joblib"
        joblib.dump(self.best_model, model_path)

        # Save metadata with enhanced information
        metadata = {
            "model_name": self.best_model_name,
            "threshold": self.optimal_threshold,
            "metrics": self.metrics_dict.get(self.best_model_name, {}),
            "feature_names": self.feature_names,
            "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preprocessing": self.preprocessor,  # Include preprocessing info
        }

        # Add top features if available
        if hasattr(self, "top_features") and self.top_features:
            metadata["top_features"] = self.top_features

        # Save all models metrics for comparison
        all_metrics = {}
        for model_name, metrics in self.metrics_dict.items():
            # Extract key metrics only
            key_metrics = {
                k: v
                for k, v in metrics.items()
                if k
                in [
                    "precision",
                    "recall",
                    "f1",
                    "auc_roc",
                    "business_metric",
                    "threshold",
                ]
            }
            all_metrics[model_name] = key_metrics

        metadata["all_models_metrics"] = all_metrics

        metadata_path = save_dir / "model_metadata.joblib"
        joblib.dump(metadata, metadata_path)

        logger.info(f"Best model ({self.best_model_name}) saved to {model_path}")
        logger.info(f"Model metadata saved to {metadata_path}")

        return model_path, metadata_path

    def plot_model_comparison(self) -> None:
        """Plot comparison of model performance metrics."""
        if not self.metrics_dict:
            logger.warning(
                "No metrics available for comparison. Run train_and_evaluate_all first."
            )
            return

        # Create comparison dataframe
        comparison_df = pd.DataFrame(self.metrics_dict).T

        # Plot metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()

        metrics_to_plot = ["precision", "recall", "f1", "auc_roc"]
        titles = ["Precision", "Recall", "F1 Score", "AUC-ROC"]

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
        plt.savefig(MODELS_DIR / "model_comparison.png")
        logger.info(
            f"Model comparison plot saved to {MODELS_DIR / 'model_comparison.png'}"
        )


def load_processed_data(use_advanced=True):
    """Load preprocessed data with optional advanced features.

    Args:
        use_advanced: Whether to use advanced processed data (True) or baseline data (False)

    Returns:
        Tuple of datasets and preprocessor
    """
    try:
        if use_advanced:
            # Load advanced processed data
            logger.info("Loading advanced processed data...")
            X_train = np.load(PROCESSED_DATA_DIR / "X_train.npy")
            X_test = np.load(PROCESSED_DATA_DIR / "X_test.npy")
            y_train = np.load(PROCESSED_DATA_DIR / "y_train.npy")
            y_test = np.load(PROCESSED_DATA_DIR / "y_test.npy")

            # Load preprocessor if available
            try:
                preprocessor = joblib.load(PROCESSED_DATA_DIR / "preprocessor.joblib")
                logger.info("Loaded preprocessor with advanced features")

                # Load feature names from CSV if available
                try:
                    feature_names_df = pd.read_csv(
                        PROCESSED_DATA_DIR / "feature_names.csv"
                    )
                    feature_names = feature_names_df["feature_name"].tolist()
                    logger.info(f"Loaded {len(feature_names)} feature names")
                except FileNotFoundError:
                    logger.warning("Feature names file not found, using default names")
                    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

                return X_train, X_test, y_train, y_test, preprocessor, feature_names

            except FileNotFoundError:
                logger.warning("Preprocessor not found, returning data only")
                return X_train, X_test, y_train, y_test, None, None

        else:
            # Load baseline data
            logger.info("Loading baseline processed data...")
            baseline_dir = PROCESSED_DATA_DIR / "baseline"

            if baseline_dir.exists():
                X_train = np.load(baseline_dir / "X_train.npy")
                X_test = np.load(baseline_dir / "X_test.npy")
                y_train = np.load(baseline_dir / "y_train.npy")
                y_test = np.load(baseline_dir / "y_test.npy")
                logger.info("Loaded baseline data without advanced features")

                # Use default feature names for baseline data
                feature_names = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
                return X_train, X_test, y_train, y_test, None, feature_names
            else:
                logger.warning(
                    "Baseline directory not found, falling back to advanced data"
                )
                # Recursively call with advanced=True
                return load_processed_data(use_advanced=True)

    except Exception as e:
        logger.error(f"Error loading processed data: {str(e)}")
        raise


def visualize_model_results(model, X_test, y_test, model_name, threshold=0.5):
    """Visualize model results with confusion matrix and ROC curve.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
        threshold: Classification threshold
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(MODELS_DIR / f"{model_name}_confusion_matrix.png")

    # ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(MODELS_DIR / f"{model_name}_roc_curve.png")

    # Log results
    logger.info(f"Visualizations for {model_name} saved to {MODELS_DIR}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    logger.info(f"AUC-ROC Score: {roc_auc:.4f}")


if __name__ == "__main__":
    logger.info("Starting model training with advanced methods")

    # Load data - use advanced features by default
    # Load both advanced and baseline data for comparison
    try:
        logger.info("Loading advanced processed data...")
        advanced_data = load_processed_data(use_advanced=True)

        if len(advanced_data) >= 4:
            # Unpack results - may include preprocessor and feature_names
            if len(advanced_data) == 6:
                (
                    X_train_adv,
                    X_test_adv,
                    y_train_adv,
                    y_test_adv,
                    preprocessor,
                    feature_names,
                ) = advanced_data
            else:
                X_train_adv, X_test_adv, y_train_adv, y_test_adv = advanced_data[:4]
                preprocessor, feature_names = None, None

            logger.info(
                f"Loaded advanced dataset with {X_train_adv.shape[0]} samples and {X_train_adv.shape[1]} features"
            )

            # Try to load baseline data for comparison
            try:
                logger.info("Loading baseline processed data for comparison...")
                baseline_data = load_processed_data(use_advanced=False)

                if len(baseline_data) >= 4:
                    X_train_base, X_test_base, y_train_base, y_test_base = (
                        baseline_data[:4]
                    )
                    logger.info(
                        f"Loaded baseline dataset with {X_train_base.shape[0]} samples and {X_train_base.shape[1]} features"
                    )
                    run_comparison = True
                else:
                    run_comparison = False
            except Exception:
                logger.warning("Could not load baseline data, skipping comparison")
                run_comparison = False

            # Default to using advanced data
            X_train, X_test, y_train, y_test = (
                X_train_adv,
                X_test_adv,
                y_train_adv,
                y_test_adv,
            )

        else:
            raise ValueError("Insufficient data loaded")

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

    # Set up for model training
    logger.info("Initializing model trainer with advanced models")
    trainer = ModelTrainer(advanced_models=True)

    # Set preprocessor for model metadata
    if preprocessor is not None:
        trainer.preprocessor = preprocessor

    # Set feature names for interpretability
    if feature_names is not None:
        trainer.feature_names = feature_names
    else:
        # Fallback to default feature names
        trainer.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    logger.info(
        f"Training and evaluating models using {PRIMARY_METRIC} as primary metric"
    )

    # Train and evaluate all models
    results = trainer.train_and_evaluate_all(
        X_train,
        y_train,
        X_test,
        y_test,
        optimize_threshold=True,
        best_metric=PRIMARY_METRIC,
    )

    # Run comparison with baseline if available
    if run_comparison:
        logger.info("Running comparison with baseline data")
        baseline_trainer = ModelTrainer(advanced_models=False)
        baseline_trainer.feature_names = [f"V{i}" for i in range(1, 29)] + [
            "Time",
            "Amount",
        ]

        # Just train a few models for baseline comparison
        baseline_models = {
            "logistic_regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            "random_forest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            "xgboost": xgb.XGBClassifier(**XGBOOST_PARAMS),
        }
        baseline_trainer.models = baseline_models

        baseline_results = baseline_trainer.train_and_evaluate_all(
            X_train_base,
            y_train_base,
            X_test_base,
            y_test_base,
            optimize_threshold=True,
            best_metric=PRIMARY_METRIC,
        )

        # Log comparison
        logger.info("Baseline vs Advanced Model Comparison:")
        for model_name in baseline_models:
            if (
                model_name in trainer.metrics_dict
                and model_name in baseline_trainer.metrics_dict
            ):
                adv_f1 = trainer.metrics_dict[model_name].get("f1", 0)
                base_f1 = baseline_trainer.metrics_dict[model_name].get("f1", 0)
                improvement = (adv_f1 - base_f1) / base_f1 * 100 if base_f1 > 0 else 0
                logger.info(
                    f"  {model_name}: Advanced F1={adv_f1:.4f}, Baseline F1={base_f1:.4f}, Improvement: {improvement:.2f}%"
                )

    # Advanced hyperparameter tuning with more options
    best_model_name = trainer.best_model_name
    tuning_successful = False

    try:
        logger.info(f"Performing advanced hyperparameter tuning for {best_model_name}")

        if (
            "xgboost" in best_model_name
            or "gradient_boosting" in best_model_name
            or "lightgbm" in best_model_name
        ):
            # More extensive tuning for tree-based models
            param_dist = {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [3, 4, 5, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5] if "xgboost" in best_model_name else [],
                "gamma": [0, 0.1, 0.2] if "xgboost" in best_model_name else [],
                # Use only the faster histogram method for XGBoost
                "tree_method": ["hist"] if "xgboost" in best_model_name else [],
            }
            trainer.tune_hyperparameters(
                best_model_name, param_dist, X_train, y_train, n_iter=20
            )
            tuning_successful = True

        elif "random_forest" in best_model_name:
            param_dist = {
                "n_estimators": [100, 200, 300, 500],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "class_weight": ["balanced", "balanced_subsample", None],
            }
            trainer.tune_hyperparameters(
                best_model_name, param_dist, X_train, y_train, n_iter=20
            )
            tuning_successful = True

        elif "logistic_regression" in best_model_name:
            param_dist = {
                "C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2", "l1", "elasticnet", None],
                "solver": ["lbfgs", "newton-cg", "liblinear", "saga"],
                "class_weight": [None, "balanced"],
                "max_iter": [100, 200, 500, 1000],
                "l1_ratio": [0.1, 0.5, 0.9],  # For elasticnet
            }
            trainer.tune_hyperparameters(
                best_model_name, param_dist, X_train, y_train, n_iter=20
            )
            tuning_successful = True

        elif "neural_network" in best_model_name:
            param_dist = {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                "activation": ["relu", "tanh", "logistic"],
                "solver": ["adam", "sgd", "lbfgs"],
                "alpha": [0.0001, 0.001, 0.01, 0.1],
                "learning_rate": ["constant", "adaptive", "invscaling"],
                "max_iter": [200, 500, 1000],
            }
            trainer.tune_hyperparameters(
                best_model_name, param_dist, X_train, y_train, n_iter=15
            )
            tuning_successful = True

        elif (
            "voting_ensemble" in best_model_name
            or "stacking_ensemble" in best_model_name
        ):
            logger.info(
                f"Skipping hyperparameter tuning for {best_model_name} - ensemble models already optimized"
            )

        else:
            logger.info(f"No specific tuning parameters defined for {best_model_name}")

    except Exception as e:
        logger.warning(f"Hyperparameter tuning failed: {str(e)}. Skipping this step.")

    # Re-evaluate all models after tuning
    if tuning_successful:
        logger.info(f"Re-evaluating tuned {best_model_name}")
        metrics = trainer.evaluate_model(
            best_model_name,
            X_test,
            y_test,
            trainer.optimal_threshold,
            business_cost_ratio=BUSINESS_COST_RATIO,
        )
        trainer.metrics_dict[f"{best_model_name}_tuned"] = metrics

        # Check if tuning improved the model
        if metrics[PRIMARY_METRIC] > trainer.best_score:
            logger.info(
                f"Tuning improved {PRIMARY_METRIC} from {trainer.best_score:.4f} to {metrics[PRIMARY_METRIC]:.4f}"
            )
            trainer.best_score = metrics[PRIMARY_METRIC]
        else:
            logger.info(
                f"Tuning did not improve {PRIMARY_METRIC} (before: {trainer.best_score:.4f}, after: {metrics[PRIMARY_METRIC]:.4f})"
            )
    else:
        logger.info(f"Using best untuned model {best_model_name}")
        metrics = trainer.metrics_dict[best_model_name]

    # Generate model explanations for better interpretability
    try:
        # Generate SHAP explanations for the best model
        logger.info("Generating model explanations with SHAP")
        trainer.generate_model_explanations(X_test)

        # Generate standard visualizations
        trainer.plot_model_comparison()
        visualize_model_results(
            trainer.best_model,
            X_test,
            y_test,
            trainer.best_model_name,
            trainer.optimal_threshold,
        )
    except Exception as e:
        logger.warning(f"Error generating visualizations: {str(e)}")

    # Save the best model with all metadata
    try:
        model_path, metadata_path = trainer.save_best_model()
        logger.info("Model training completed successfully")
        logger.info(
            f"Best model: {trainer.best_model_name} with {PRIMARY_METRIC} score: {metrics.get(PRIMARY_METRIC, 'N/A')}"
        )
        logger.info(
            f"F1 Score: {metrics.get('f1', 'N/A')}, Precision: {metrics.get('precision', 'N/A')}, Recall: {metrics.get('recall', 'N/A')}"
        )
        logger.info(f"Optimal threshold: {trainer.optimal_threshold:.4f}")

        # Print top features if available
        if hasattr(trainer, "top_features") and trainer.top_features:
            logger.info(
                f"Top predictive features: {', '.join(trainer.top_features[:5])}"
            )

    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
