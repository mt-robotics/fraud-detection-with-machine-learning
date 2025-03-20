#!/bin/bash
set -e

# Print banner
echo "=================================================="
echo "      Fraud Detection Project Initialization       "
echo "=================================================="

# Create directories if not exist
echo "Creating project directory structure..."
mkdir -p data/raw data/processed models logs

# Create empty README files for git
echo "Creating placeholder files for git..."
touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep logs/.gitkeep

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    # git config user.name "Your Name"
    # git config user.email "your.email@example.com"
    # echo "Please update your git user information in .git/config"
else
    echo "Git repository already initialized"
fi

# # Add files to git
# echo "Adding files to git..."
# git add .gitignore README.md requirements.txt

# # First commit
# echo "Creating initial commit..."
# git commit -m "Initial project structure"

echo -e "\nProject initialized successfully!"
echo "Next steps:"
echo "1. Add the Credit Card Fraud Detection dataset to data/raw/"
echo "2. Run the exploration notebook: jupyter notebook notebooks/01_data_exploration.ipynb"
echo "3. Train models: ./scripts/train_and_evaluate.sh"
echo "=================================================="