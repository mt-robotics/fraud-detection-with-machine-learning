# Raw Data Directory

This directory is intended for storing raw, unprocessed data files for the fraud detection project.

## Expected Files

- `creditcard.csv` - The Credit Card Fraud Detection dataset from Kaggle

## How to Download the Dataset

1. Register on Kaggle and get your API credentials
2. Set up your Kaggle API credentials by creating a `~/.kaggle/kaggle.json` file
3. Run the download script:

```bash
python src/data/download_data.py
```

Alternatively, you can download the dataset directly from Kaggle:
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Data Description

The dataset contains transactions made by credit cards in September 2013 by European cardholders.
It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

### Features:

- `Time`: Seconds elapsed between each transaction and the first transaction
- `Amount`: Transaction amount
- `V1-V28`: Principal components obtained with PCA transformation (anonymized features)
- `Class`: Target variable - 1 for fraudulent transactions, 0 for legitimate ones

## Note

Do not modify the raw data files. Use the data preprocessing pipeline to clean and transform the data.