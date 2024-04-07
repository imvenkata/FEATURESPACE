POS_ENTRY_MODE_MAP = {
    0: "Entry Mode Unknown",
    1: "POS Entry Mode Manual",
    2: "POS Entry Model Partial MSG Stripe",
    5: "POS Entry Circuit Card",
    7: "RFID Chip (Chip card processed using chip)",
    79: "POS Entry Mode Unknown",
    80: "Chip Fallback to Magnetic Stripe",
    81: "POS Entry E-Commerce",
    90: "POS Entry Full Magnetic Stripe Read",
    91: "POS Entry Circuit Card Partial",
}

DATA_DIR = "/Users/venkata.medabala/Projects/ml/featurespace/data"

EVENT_ID = ["eventId"]
FEATURES = [
    "mcc",
    "posEntryMode",
    "availableCash",
    "transactionAmount",
    # "amount_binned",
    "transaction_hour",
    "transaction_day",
    "time_since_last_transaction",
    "amount_to_available_cash_ratio",
    "amount_deviation_from_account_avg",
    "transactions_per_account",
    "average_transaction_amount_per_account",
    "std_transaction_amount_per_account",
    "is_new_merchant",
    "merchent_zip3_country",
    "merchantCountry",
    # "merchant_risk_score",
    # "zip_code_risk_score",
    # "country_risk_score",
]

LABEL = ["is_fraud"]


AMOUNT_BIN_DICT = {
    "low": 0,
    "medium": 25,
    "high": 75,
    "very_high": 150,
    "extreme": 999999999,
}

TEST_DATA_RATIO = 0.05

CATEGORICAL_FEATURES = [
    "mcc",
    "posEntryMode",
    "merchent_zip3_country",
    # "amount_binned",
    "merchantCountry",
]

SMOTE_ENN_PARAMS = {"sampling_strategy": "minority"}

CATBOOST_PARAM_GRID = {
    "iterations": [25, 500],
    "learning_rate": [0.05, 0.1],
    "depth": [6, 8],
    "l2_leaf_reg": [1, 3],
    "class_weights": [[1, 10]],
}
