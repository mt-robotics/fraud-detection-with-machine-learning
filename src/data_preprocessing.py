"""Module for data preprocessing functions with advanced feature engineering."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from scipy import stats
from loguru import logger

# Add the project root directory to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# pylint: disable=wrong-import-position
from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    TARGET_COLUMN,
)


def load_data() -> pd.DataFrame:
    """Load raw credit card fraud data."""
    try:
        data_path = RAW_DATA_DIR / "creditcard.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def add_interaction_features(X_df):
    """Add interaction features between important variables."""

    # Important numerical features based on domain knowledge and EDA
    important_features = ["V4", "V14", "V10", "V12", "V17"]

    # Create interaction terms
    for i, feat1 in enumerate(important_features):
        for feat2 in important_features[i + 1 :]:
            # Multiplication interaction
            X_df[f"{feat1}_{feat2}_mult"] = X_df[feat1] * X_df[feat2]
            # Ratio interaction (with safeguards against division by zero)
            X_df[f"{feat1}_{feat2}_ratio"] = X_df[feat1] / (X_df[feat2] + 1e-8)

    # Add polynomial features for high importance features
    for feat in important_features:
        X_df[f"{feat}_squared"] = X_df[feat] ** 2

    # Create time-based features
    X_df["time_sin"] = np.sin(
        2 * np.pi * X_df["Time"] / 86400
    )  # 86400 seconds in a day
    X_df["time_cos"] = np.cos(2 * np.pi * X_df["Time"] / 86400)

    # Amount-related features
    X_df["log_amount"] = np.log1p(X_df["Amount"])

    return X_df


def outlier_detection(X):
    """Detect and handle outliers in the dataset."""
    X_outlier = X.copy()

    # Z-score based outlier detection
    z_scores = np.abs(stats.zscore(X_outlier, nan_policy="omit"))
    filtered_entries = (z_scores < 5).all(axis=1)

    # Cap outliers rather than removing them (better for fraud detection)
    for col in X_outlier.columns:
        upper_limit = X_outlier[col].mean() + 5 * X_outlier[col].std()
        lower_limit = X_outlier[col].mean() - 5 * X_outlier[col].std()
        X_outlier[col] = np.clip(X_outlier[col], lower_limit, upper_limit)

    logger.info(
        f"Outlier handling completed. Capped {(~filtered_entries).sum()} outlier entries."
    )
    return X_outlier


def feature_selection(X_train, y_train, threshold=0.99):
    """Select most important features using an ensemble method."""
    # Use Random Forest for feature importance
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        threshold="median",
    )
    selector.fit(X_train, y_train)
    support = selector.get_support()

    # Get selected feature names
    selected_features = X_train.columns[support].tolist()

    logger.info(f"Selected {len(selected_features)} features out of {X_train.shape[1]}")
    return selected_features, selector


def preprocess_data(
    df: pd.DataFrame,
    balance_data: bool = True,
    advanced_features: bool = True,
    resampling_method: str = "smote_tomek",
) -> tuple:
    """Preprocess the data for model training with advanced feature engineering.

    Args:
        df: Raw dataframe
        balance_data: Whether to apply resampling to handle class imbalance
        advanced_features: Whether to apply advanced feature engineering
        resampling_method: Method to use for resampling ('smote', 'adasyn', 'smote_tomek')

    Returns:
        Tuple containing processed train/test data and preprocessing objects
    """
    try:
        # Convert to DataFrame for feature engineering
        df_copy = df.copy()

        # Separate features and target
        X = df_copy.drop(TARGET_COLUMN, axis=1)
        y = df_copy[TARGET_COLUMN]

        # Split data before advanced preprocessing to avoid data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        # Convert to DataFrame for feature engineering
        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)

        # Apply advanced feature engineering if requested
        preprocessor = {}

        if advanced_features:
            logger.info("Applying advanced feature engineering...")

            # Handle outliers (fraud cases are outliers by nature, so be careful)
            X_train_df = outlier_detection(X_train_df)

            # Add interaction and domain-specific features
            X_train_df = add_interaction_features(X_train_df)
            X_test_df = add_interaction_features(X_test_df)

            # Apply power transform for normalization (better than StandardScaler for skewed data)
            power_transformer = PowerTransformer(method="yeo-johnson", standardize=True)
            X_train_array = power_transformer.fit_transform(X_train_df)
            X_test_array = power_transformer.transform(X_test_df)
            preprocessor["power_transformer"] = power_transformer

            # Convert back to DataFrame for feature selection
            X_train_df = pd.DataFrame(X_train_array, columns=X_train_df.columns)
            X_test_df = pd.DataFrame(X_test_array, columns=X_test_df.columns)

            # Apply feature selection to remove less important features
            selected_features, selector = feature_selection(X_train_df, y_train)
            preprocessor["selector"] = selector
            preprocessor["selected_features"] = selected_features

            # Apply dimensionality reduction with PCA for visualization
            if len(selected_features) > 10:
                pca = PCA(n_components=10, random_state=RANDOM_STATE)
                pca_features = pca.fit_transform(X_train_df[selected_features])
                X_train_df_pca = pd.DataFrame(
                    pca_features,
                    columns=[f"PCA{i+1}" for i in range(pca.n_components_)],
                )

                # Add PCA features to the dataset
                for col in X_train_df_pca.columns:
                    X_train_df[col] = X_train_df_pca[col].values

                # Transform test data
                pca_features_test = pca.transform(X_test_df[selected_features])
                X_test_df_pca = pd.DataFrame(
                    pca_features_test,
                    columns=[f"PCA{i+1}" for i in range(pca.n_components_)],
                )

                # Add PCA features to test data
                for col in X_test_df_pca.columns:
                    X_test_df[col] = X_test_df_pca[col].values

                preprocessor["pca"] = pca
        else:
            # Apply standard scaling if no advanced features
            scaler = StandardScaler()
            X_train_array = scaler.fit_transform(X_train_df)
            X_test_array = scaler.transform(X_test_df)
            preprocessor["scaler"] = scaler

        # Convert back to arrays for model training
        X_train_array = X_train_df.values
        X_test_array = X_test_df.values

        # Feature names for later interpretation
        preprocessor["feature_names"] = X_train_df.columns.tolist()

        # Convert targets to numpy arrays to ensure consistent type
        y_train_np = np.array(y_train)
        y_test_np = np.array(y_test)

        # Handle class imbalance if requested
        if balance_data:
            if resampling_method == "smote":
                resampler = SMOTE(random_state=RANDOM_STATE)
                X_train_balanced, y_train_balanced = resampler.fit_resample(
                    X_train_array, y_train_np
                )
                logger.info("Applied SMOTE to balance training data")

            elif resampling_method == "adasyn":
                resampler = ADASYN(random_state=RANDOM_STATE)
                X_train_balanced, y_train_balanced = resampler.fit_resample(
                    X_train_array, y_train_np
                )
                logger.info("Applied ADASYN to balance training data")

            elif resampling_method == "smote_tomek":
                resampler = SMOTETomek(random_state=RANDOM_STATE)
                X_train_balanced, y_train_balanced = resampler.fit_resample(
                    X_train_array, y_train_np
                )
                logger.info("Applied SMOTETomek to balance training data")

            else:
                raise ValueError(f"Unknown resampling method: {resampling_method}")

            preprocessor["resampler"] = resampler
            return (
                X_train_balanced,
                X_test_array,
                np.array(y_train_balanced),
                y_test_np,
                preprocessor,
            )

        return (X_train_array, X_test_array, y_train_np, y_test_np, preprocessor)

    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise


def save_processed_data(X_train, X_test, y_train, y_test, preprocessor=None):
    """Save processed data and preprocessing objects to files."""
    try:
        np.save(PROCESSED_DATA_DIR / "X_train.npy", X_train)
        np.save(PROCESSED_DATA_DIR / "X_test.npy", X_test)
        np.save(PROCESSED_DATA_DIR / "y_train.npy", y_train)
        np.save(PROCESSED_DATA_DIR / "y_test.npy", y_test)

        # Save preprocessor if provided
        if preprocessor is not None:
            import joblib

            joblib.dump(preprocessor, PROCESSED_DATA_DIR / "preprocessor.joblib")

            # Save feature names for interpretability
            if "feature_names" in preprocessor:
                feature_names = preprocessor["feature_names"]
                pd.Series(feature_names).to_csv(
                    PROCESSED_DATA_DIR / "feature_names.csv",
                    index=False,
                    header=["feature_name"],
                )

        logger.info("Saved processed data to files")
    except Exception as e:
        logger.error(f"Error saving processed data: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("Starting data preprocessing with advanced feature engineering")
    df = load_data()

    # Process data with different resampling methods for comparison
    resampling_methods = ["smote", "adasyn", "smote_tomek"]

    # Default advanced method
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        df,
        balance_data=True,
        advanced_features=True,
        resampling_method="smote_tomek",  # More sophisticated resampling
    )

    # Save the advanced preprocessed data
    save_processed_data(X_train, X_test, y_train, y_test, preprocessor)

    # Also save a version with basic preprocessing for comparison
    logger.info("Creating baseline dataset with standard preprocessing for comparison")
    X_train_basic, X_test_basic, y_train_basic, y_test_basic, preprocessor_basic = (
        preprocess_data(
            df,
            balance_data=True,
            advanced_features=False,  # No advanced features
            resampling_method="smote",  # Standard SMOTE
        )
    )

    # Save the basic preprocessed data with a different name
    BASIC_DIR = PROCESSED_DATA_DIR / "baseline"
    BASIC_DIR.mkdir(exist_ok=True)
    np.save(BASIC_DIR / "X_train.npy", X_train_basic)
    np.save(BASIC_DIR / "X_test.npy", X_test_basic)
    np.save(BASIC_DIR / "y_train.npy", y_train_basic)
    np.save(BASIC_DIR / "y_test.npy", y_test_basic)

    # Log information about the datasets
    logger.info(
        f"Advanced dataset shape: X_train={X_train.shape}, y_train={y_train.shape}"
    )
    logger.info(f"Test dataset shape: X_test={X_test.shape}, y_test={y_test.shape}")
    logger.info(
        f"Generated {len(preprocessor.get('feature_names', []))} features in advanced dataset"
    )

    logger.info("Data preprocessing completed")
