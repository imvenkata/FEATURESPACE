import pandas as pd
from typing import Dict, List
from src.config import AMOUNT_BIN_DICT
from datetime import datetime
from pydantic import BaseModel, validator


class TransactionData(BaseModel):
    """Defines a data model for transaction data."""

    accountNumber: str
    transactionTime: datetime
    eventId: str
    merchantId: str
    mcc: str
    merchantCountry: str
    merchantZip: str
    posEntryMode: int
    availableCash: float
    transactionAmount: float

    @validator("transactionAmount")
    def check_amount_positive(cls, value):
        """Ensures transaction amount is positive."""
        if value <= 0:
            raise ValueError("Transaction amount must be positive")
        return value


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts features from the transaction data.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with additional extracted features.
    """

    df = calc_time_features(df)
    df = calc_amount_features(df)
    df = calc_account_features(df)
    df = calc_marchant_features(df)
    df = calc_zip_risk(df)
    df = one_hot_encode(df, categorical_features)
    return df


# Define the categorical features for one-hot encoding
categorical_features = [
    "mcc",
    "posEntryMode",
    "merchantCountry",
]


def calc_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts time-based features from the transaction data.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with additional time-based features.
    """

    df = _extract_time_components(df)
    df = calc_time_since_previous(df)
    return df


def calc_time_since_previous(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the time since the last transaction for each account.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with a 'time_since_last_transaction' column.
    """

    df = df.sort_values(["accountNumber", "transactionTime"])
    df["time_since_last_transaction"] = df.groupby("accountNumber")[
        "transactionTime"
    ].diff()
    return df


def _extract_time_components(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts hour, day, and month components from transactionTime.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with 'transaction_hour', 'transaction_day', 'transaction_month' columns.
    """

    df["transaction_hour"] = df["transactionTime"].dt.hour
    df["transaction_day"] = df["transactionTime"].dt.day
    df["transaction_month"] = df["transactionTime"].dt.month
    return df


def calc_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates amount-related features.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
         The DataFrame with 'amount_to_available_cash_ratio' and 'amount_binned' columns.
    """

    df["amount_to_available_cash_ratio"] = df["transactionAmount"] / df["availableCash"]
    df = bin_amounts(df)
    return df


def bin_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """Bins transaction amounts using a dictionary from the config.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with an 'amount_binned' column.
    """

    def apply_bins(amount):
        for label, upper_bound in AMOUNT_BIN_DICT.items():
            if amount <= upper_bound:
                return label

    df["amount_binned"] = df["transactionAmount"].apply(apply_bins)
    return df


def calc_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates account-based features.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with 'transactions_per_account', 'average_transaction_amount_per_account'.
    """

    df_account_features = df.groupby("accountNumber")[["transactionAmount"]].agg(
        ["count", "mean"]
    )
    df_account_features.columns = [
        "transactions_per_account",
        "average_transaction_amount_per_account",
    ]
    df = df.join(df_account_features, on="accountNumber")
    return df


def calc_marchant_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates merchant-related features for fraud detection.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with additional merchant-based features, including
        'is_new_merchant' and 'merchant_risk_score'.
    """

    df = track_new_merchants(df)
    df = calc_merchant_risk(df)
    return df


# Is New Merchant Feature
def track_new_merchants(df: pd.DataFrame) -> pd.DataFrame:
    """Identifies transactions associated with new merchants for a given account.

    Args:
         df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with an 'is_new_merchant' column indicating whether a
        transaction is the first interaction between an account and merchant.
    """

    df["is_new_merchant"] = ~df.duplicated(
        subset=["accountNumber", "merchantId"], keep="first"
    )
    return df


# Merchant Risk Score
def calc_merchant_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates a risk score for each merchant based on historical fraud patterns.

    Args:
        df: The DataFrame containing transaction data, including an 'is_fraud' column.

    Returns:
        The DataFrame with a 'merchant_risk_score' column, representing the likelihood
        of fraud associated with each merchant.
    """

    event_by_merchant = (
        df[df["is_fraud"] == 1].groupby("merchantId")["transactionAmount"].count()
    )
    total_by_merchant = df.groupby("merchantId")["transactionAmount"].count()
    merchant_risk = event_by_merchant / total_by_merchant
    merchant_risk.index = merchant_risk.index.astype(str)
    df["merchant_risk_score"] = df["merchantId"].map(merchant_risk).fillna(0.0)
    return df


def calc_zip_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates risk scores for each merchant zip code based on fraud rates.

    Args:
        df: The DataFrame containing transaction data, including 'merchantZip_imputed' and 'is_fraud' columns.

    Returns:
        The DataFrame with a 'zip_code_risk_score' column.
    """

    # Calculate the risk scores
    zip_risk_map = _zip_risk_score(df)

    # Add risk score as a new column
    df["zip_code_risk_score"] = df["merchantZip_imputed"].map(zip_risk_map)
    return df


def _zip_risk_score(df: pd.DataFrame) -> Dict[str, float]:
    """Calculates risk scores for merchant zip codes (helper function).

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        A dictionary mapping merchant zip codes to their calculated risk scores.
    """

    fraud_by_zip = df[df["is_fraud"] == 1].groupby("merchantZip_imputed").size()
    total_by_zip = df.groupby("merchantZip_imputed").size()
    zip_risk_scores = fraud_by_zip / total_by_zip
    zip_risk_scores = zip_risk_scores.fillna(0.0)
    return zip_risk_scores.to_dict()


def one_hot_encode(df, columns):
    """Performs one-hot encoding on specified columns, dropping the original columns.

    Args:
        df: The DataFrame containing the columns to encode.
        columns: A list of column names to be one-hot encoded.

    Returns:
        The DataFrame with one-hot encoded features.
    """

    df = pd.get_dummies(df, columns=columns, drop_first=True)
    return df
