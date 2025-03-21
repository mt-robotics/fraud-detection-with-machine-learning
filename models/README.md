# Models Directory

This directory stores trained machine learning models and related artifacts.

## Expected Files

After training, you should see the following files:

- `best_model.joblib` - The best performing model serialized using joblib
- `model_metadata.joblib` - Model metadata including threshold, feature importance, etc.
- Various model visualizations (PNG files)

## Model Registry

For a production environment, consider implementing a more sophisticated model registry that tracks:

- Model versions
- Performance metrics
- Training data lineage
- Model artifacts
- Deployment status

## How to Generate Model Files

Run the model training script:

```bash
python src/model_training.py
```

Or use the convenience script:

```bash
./scripts/train_and_evaluate.sh
```

## Note

This directory is kept in version control with `.gitkeep`, but the actual model files should be added to `.gitignore` to avoid committing large binary files.