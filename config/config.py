"""Configuration settings for the fraud detection project."""

from pathlib import Path
import os
from dotenv import load_dotenv, dotenv_values

# Base paths - defined early so we can use it for env file path
BASE_DIR = Path(__file__).parent.parent

# Load environment variables
env = os.getenv("ENV", "development")  # Default to development
env_file = BASE_DIR / f".env.{env}"  # Path to a specific .env file
print(f"Loading environment: {env} from {env_file}")

if not env_file.exists():
    print(f"Error: {env_file} does not exist!")
else:
    print(f"{env_file} exists. Loading...")
    load_dotenv(dotenv_path=env_file)  # Explicitly specify dotenv_path parameter

# # Print all environment variables for debugging
# print("Loaded environment variables:")
# for key, value in os.environ.items():
#     print(f"{key}: {value}")

# Print only the variables from the .env file
env_vars = dotenv_values(dotenv_path=env_file)
print("Variables loaded from .env file:")
for key, value in env_vars.items():
    print(f"{key}: {value}")

# Define other path directories based on BASE_DIR
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "Class"

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv(
    "LOG_FORMAT",
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# Print LOG_LEVEL to verify
print(f"\nLOG_LEVEL: {LOG_LEVEL}")
print(f"\nEnvironment currently active: {env}")
