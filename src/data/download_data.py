"""Script to download and verify the credit card fraud dataset."""

import os
import subprocess
from loguru import logger
from config.config import RAW_DATA_DIR


def download_dataset():
    """Download the credit card fraud dataset from Kaggle."""
    try:
        # Ensure Kaggle API credentials are set up
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            raise FileNotFoundError(
                "Kaggle API credentials not found. Please follow the setup instructions in README.md"
            )

        dataset_name = "mlg-ulb/creditcardfraud"
        output_dir = RAW_DATA_DIR

        logger.info(f"Downloading dataset from Kaggle: {dataset_name}")

        # Use subprocess to run kaggle command instead of importing the library
        cmd = [
            "kaggle",
            "datasets",
            "download",
            dataset_name,
            "--path",
            str(output_dir),
            "--unzip",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.returncode == 0:
            logger.info(f"Dataset downloaded successfully to {output_dir}")
        else:
            logger.error(f"Download failed: {result.stderr}")
            raise Exception(result.stderr)

    except FileNotFoundError:
        logger.error("Kaggle CLI not found. Make sure kaggle is installed.")
        logger.info("You can install it with: pip install kaggle")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"Kaggle command failed: {e.stderr}")
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise


if __name__ == "__main__":
    download_dataset()
