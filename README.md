**Transaction Fraud Detection model**

This project focuses on building a robust fraud detection model using the CatBoost algorithm.

**Data:**

Transactional Data

- transactionTime: The date and time the transaction was requested.
- eventId: A unique string identifier for the transaction.
- accountNumber: The account number associated with the transaction.
- merchantId: A unique string identifier for the merchant.
- mcc: Merchant Category Code (represents the merchant's business type).
- merchantCountry: The country where the merchant is located.
- merchantZip: Truncated/partial zip code of the merchant's location.
- posEntryMode: Point of Sale entry mode (e.g., manual, chip card, e-commerce).
- transactionAmount: The value of the transaction (presumably in GBP).
- availableCash: Rounded amount available in the account prior to the transaction.

Events data

- is_fraud: This sis the target variable to be used in the model building. 0 indicates normal transaction, 1 indicate fraud
- reportedTime: The time the transaction was reported (assuming different from 'transactionTime').

Derived / Calculated Features

- transaction_hour: Hour extracted from 'transactionTime'.
- transaction_day: Day extracted from 'transactionTime'.
- transaction_month: Month extracted from 'transactionTime'.
- time_since_last_transaction: Time difference between the current and previous transaction for the same account.
- amount_to_available_cash_ratio: Ratio of the transaction amount to the available cash before the transaction.
- amount_binned: Category of the transaction amount (e.g., low, medium, high).
- transactions_per_account: Count of transactions associated with each account.
- average_transaction_amount_per_account: Average transaction amount for each account.
- is_new_merchant: Indicates if the transaction is the first interaction between the account and merchant.
- merchant_risk_score: A calculated risk score for each merchant, potentially based on fraud rates.
- zip_code_risk_score: A calculated risk score for each merchant zipcode, potentially based on fraud rates.

Planed but note implemented:

- distance between transaction location and the previous transaction location (approximate distance based on zipcodes)
- Moving average of transaction frequency from the same account
- Moving average of location difference between consecutive transactions from the same account
- Moving standard deviation of transaction volume from the same account
- Feature hashing for categorical data

Research Notes:

- Feature Engineering
- Model: CatBoost
- Data Splitting:
  - Out-of-sample test set (5%) for final evaluation.
  - Development set stratified split into train (85%) and validation (15%).
- Imbalance Handling: SMOTEC technique.
- Missing Value Imputation: Merchant zip codes imputed based on previous transactions when available.

Note Implemented yet:

- Experiment tracking
- Scoring pipeline
- Data drift analysis
- unit tests

**Directory Structure**

```
├── README.md
├── data
│ ├── precal
│ ├── processed
│ ├── raw
├── models
│ └── catboost_fraud_model.joblib
├── notebooks
│ └── example_usage.ipynb
├── poetry.lock
├── pyproject.toml
├── requirements.txt
├── setup.py
├── src
│ ├── config.py
│ ├── data
│ ├── features
│ ├── main.py
│ ├── tests
│ └── train

```

## Setup

- Install Python version 3.11.1

  ```
  pyenv install
  ```

- This will download and install Python **3.11.1** which is specified in the `.project-version` file which in turn is created by the command `pyenv local 3.11.1`. This use of pyenv ensures the pinning and usage of the specified Python version.

- Create a virtualenv:

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

  This virtualenv now has the version of Python which was set by pyenv and the .project-version file.

- Install all packages for local work

  ```bash
  pip3 install -U pip wheel setuptools
  pip3 install -r requirements.txt
  ```

**Model training script**

- Update config.py file with required data directory where the csv files are store. Please feel free to play with grid search parameter during the training pipeline

- Run the training pipeline in the terminal

  ```bash
  python src/main.py
  ```

This will trigger the data preparation, feature engineering and model training sequentially. The end of the program, it find the best model based on the grid search parameters provided in config.py file. And evalauate the model performance on training, validation & test data accordingly
