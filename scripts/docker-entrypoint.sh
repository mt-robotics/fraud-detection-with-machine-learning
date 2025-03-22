#!/bin/bash
set -e

# Print banner
echo "=================================================="
echo "   Fraud Detection API - Docker Initialization    "
echo "=================================================="

# Set default environment if not provided
if [ -z "$ENV" ]; then
    echo "ENV variable not set, defaulting to development"
    export ENV=development
fi

echo "Active environment: $ENV"
echo "Using environment file: .env.$ENV"

# Check if model exists
if [ ! -f "/app/models/best_model.joblib" ]; then
    echo "Model file not found. Checking if dataset is available..."
    
    # Check if dataset exists
    if [ ! -f "/app/data/raw/creditcard.csv" ]; then
        echo "Dataset not found. Please mount a volume with the dataset at /app/data/raw/creditcard.csv"
        echo "Starting API in demo mode with a pre-trained model..."
        
        # Copy demo model if available
        if [ -f "/app/models/demo/best_model.joblib" ]; then
            echo "Using demo model..."
            cp /app/models/demo/best_model.joblib /app/models/
            cp /app/models/demo/model_metadata.joblib /app/models/
        else
            echo "Warning: No demo model available. API may not function correctly."
        fi
    else
        echo "Dataset found. Training new model..."
        
        # Set environment variables for non-interactive mode
        export ADVANCED_FEATURES=true
        export ENSEMBLE_MODELS=true
        
        # Run training script
        cd /app && bash scripts/train_and_evaluate.sh
        
        echo "Model training complete."
    fi
fi

# Set default value for RUN_TESTS
RUN_TESTS=${RUN_TESTS:-false}
echo "RUN_TESTS is set to: $RUN_TESTS"

# Check if we need to run tests
if [ "$RUN_TESTS" = "true" ]; then
    echo "Running test suite..."
    cd /app && python -m pytest tests/
fi

# Initialize log directory
mkdir -p /app/logs
touch /app/logs/api.log
chmod 666 /app/logs/api.log

# Print model info
if [ -f "/app/models/model_metadata.joblib" ]; then
    echo "Using model:"
    python -c "import joblib; m = joblib.load('/app/models/model_metadata.joblib'); print(f'Model: {m.get(\"model_name\", \"unknown\")}'); print(f'Performance: {m.get(\"metrics\", {}).get(\"f1\", \"unknown\")}'); print(f'Training date: {m.get(\"training_date\", \"unknown\")}');"
fi

echo "Starting Fraud Detection API..."
exec "$@"