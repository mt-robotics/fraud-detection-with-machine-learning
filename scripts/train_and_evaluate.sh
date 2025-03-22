#!/bin/bash
set -e

# Print banner
echo "=================================================="
echo "   Fraud Detection Model Training & Evaluation    "
echo "           Enhanced Performance Version           "
echo "=================================================="

# Python command
PYTHON="python"
# Check if python is available
if ! command -v $PYTHON &> /dev/null; then
    echo "Error: Python is not installed or not in PATH. Try again with python3."
    PYTHON="python3"
    if ! command -v $PYTHON &> /dev/null; then
        echo "Error: Both Python and Python3 are not installed or not in PATH."
        exit 1
    fi
fi

# Check Python version and set appropriate environment
PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d " " -f 2 | cut -d "." -f 1,2)

# Choose appropriate virtual environment based on Python version
if [ -d "venv_py311" ]; then
    echo "Activating Python 3.11 virtual environment..."
    source venv_py311/bin/activate
elif [[ "$PYTHON_VERSION" == "3.13" ]] && [ -d "venv_py311" ]; then
    echo "Python 3.13 detected. Using Python 3.11 environment for better compatibility..."
    source venv_py311/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check for required packages
echo "Checking required packages..."
$PYTHON -c "import numpy, pandas, sklearn, xgboost, lightgbm, shap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Some required packages are missing. Installing dependencies..."
    $PYTHON -m pip install -r requirements.txt
fi

# Create necessary directories
echo "Setting up directories..."
# Get the project root directory (one level up from scripts/)
PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
echo "Project root directory: $PROJECT_ROOT"
# Navigate to the project root directory
cd "$PROJECT_ROOT" || { echo "Error: Failed to navigate to project root directory."; exit 1; }
# Create the necessary directories after successfully navigating to the project root
mkdir -p data/raw data/processed models logs

# Check if dataset exists
if [ ! -f "$PROJECT_ROOT/data/raw/creditcard.csv" ]; then
    echo "Warning: Dataset not found at $PROJECT_ROOT/data/raw/creditcard.csv"
    echo "Please download the Credit Card Fraud Detection dataset from Kaggle and place it in $PROJECT_ROOT/data/raw/"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run data preprocessing with advanced feature engineering
echo -e "\n[1/4] Starting data preprocessing with advanced features..."

# Ask if user wants to run with advanced features
if [ -z "$ADVANCED_FEATURES" ]; then
    read -p "Do you want to run with advanced feature engineering? This will take longer but improve performance. (y/n, default: y) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        ADVANCED_FEATURES=true
    else
        ADVANCED_FEATURES=false
    fi
fi

# Run preprocessing
if [ "$ADVANCED_FEATURES" = true ]; then
    echo "Running with advanced feature engineering enabled..."
    $PYTHON -OO $PROJECT_ROOT/src/data_preprocessing.py
else
    echo "Running with basic preprocessing only..."
    $PYTHON -OO $PROJECT_ROOT/src/data_preprocessing.py --basic
fi

if [ $? -ne 0 ]; then
    echo "Error: Data preprocessing failed!"
    exit 1
fi
echo "Data preprocessing completed successfully!"

# Train and evaluate models with enhanced capabilities
echo -e "\n[2/4] Starting model training and evaluation..."

# Ask if user wants to run with ensemble models
if [ -z "$ENSEMBLE_MODELS" ]; then
    read -p "Do you want to include ensemble models in training? This improves performance but takes longer. (y/n, default: y) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        ENSEMBLE_MODELS=true
    else
        ENSEMBLE_MODELS=false
    fi
fi

# Add model options based on user choices
MODEL_OPTS=""
if [ "$ADVANCED_FEATURES" = true ]; then
    MODEL_OPTS="$MODEL_OPTS --advanced-features"
fi
if [ "$ENSEMBLE_MODELS" = true ]; then
    MODEL_OPTS="$MODEL_OPTS --ensemble-models"
fi

# Run model training with options
echo "Running model training with options: $MODEL_OPTS"
time $PYTHON -OO $PROJECT_ROOT/src/model_training.py $MODEL_OPTS

if [ $? -ne 0 ]; then
    echo "Error: Model training failed!"
    exit 1
fi
echo "Model training and evaluation completed successfully!"

# Run tests
echo -e "\n[3/4] Running tests..."
pytest tests/ -v
if [ $? -ne 0 ]; then
    echo "Warning: Some tests failed! Review the output above."
else
    echo "All tests passed successfully!"
fi

# Count total files in models directory
model_files=$(find $PROJECT_ROOT/models -type f | wc -l)
echo -e "\n[4/4] Checking model artifacts..."
echo "Total model files generated: $model_files"

# List key model files
echo -e "\nKey model files:"
ls -la models/best_model.joblib 2>/dev/null || echo "Warning: No best_model.joblib found!"
ls -la models/model_metadata.joblib 2>/dev/null || echo "Warning: No model_metadata.joblib found!"

# Check for SHAP visualization files
shap_files=$(find $PROJECT_ROOT/models -name "shap_*.png" | wc -l)
if [ $shap_files -gt 0 ]; then
    echo "Generated $shap_files SHAP explanation visualizations."
fi

# Check if ensemble model was used
if [ -f "$PROJECT_ROOT/models/model_metadata.joblib" ]; then
    model_type=$(grep -o '"model_name": "[^"]*"' "$PROJECT_ROOT/models/model_metadata.joblib" 2>/dev/null || echo "Unknown")
    echo "Best model type: $model_type"
fi

# Print enhanced summary
echo -e "\n=================================================="
echo "              TRAINING SUMMARY                     "
echo "=================================================="
echo "Data preprocessing: ✓ $([ "$ADVANCED_FEATURES" = true ] && echo "[Advanced]" || echo "[Basic]")"
echo "Model training:     ✓ $([ "$ENSEMBLE_MODELS" = true ] && echo "[With Ensembles]" || echo "[Standard]")"
echo "Tests:              $([ $? -eq 0 ] && echo "✓" || echo "⚠️")"
echo "Model artifacts:    $([ $model_files -gt 2 ] && echo "✓" || echo "⚠️")"
echo "Explanations:       $([ $shap_files -gt 0 ] && echo "✓" || echo "⚠️")"

# Get model performance metrics if available
if [ -f "$PROJECT_ROOT/models/model_metadata.joblib" ]; then
    f1_score=$(grep -o '"f1": [0-9.]*' "$PROJECT_ROOT/models/model_metadata.joblib" | grep -o '[0-9.]*' || echo "Unknown")
    auc_score=$(grep -o '"auc_roc": [0-9.]*' "$PROJECT_ROOT/models/model_metadata.joblib" | grep -o '[0-9.]*' || echo "Unknown")
    
    echo -e "\nPerformance metrics:"
    echo "- F1 Score:  $f1_score"
    echo "- AUC-ROC:   $auc_score"
fi

echo -e "\nNext steps:"
echo "1. Start the API: uvicorn src.api:app --host 0.0.0.0 --port 8000"
echo "2. Access the API documentation at: http://localhost:8000/docs"
echo "3. Examine model explanations in the models/ directory"
echo "=================================================="
