import pandas as pd
from typing import Dict, List
from src.config import AMOUNT_BIN_DICT
from datetime import datetime
from pydantic import BaseModel, validator
import logging
from sklearn.feature_extraction import FeatureHasher

import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    is_fraud: int

    @validator("transactionAmount")
    def check_amount_positive(cls, value):
        """Ensures transaction amount is positive."""
        if value <= 0:
            raise ValueError("Transaction amount must be positive")
        return value

    @validator("is_fraud")
    def check_is_fraud(cls, value):
        """Ensures is_fraud is either 0 or 1."""
        if value not in [0, 1]:
            raise ValueError("is_fraud must be either 0 or 1")
        return value


def extract_features(df: pd.DataFrame, save_data: bool = False) -> pd.DataFrame:
    """Orchestrates feature extraction."""

    # Define column groups
    time_cols = ["eventId", "transactionTime", "accountNumber"]
    amount_cols = ["eventId", "transactionAmount", "availableCash"]
    account_cols = ["eventId", "accountNumber", "transactionAmount", "is_fraud"]
    merchant_cols = [
        "eventId",
        "merchantId",
        "accountNumber",
        "merchantZip_imputed",
        "merchantCountry",
    ]
    # categorical_features = ["mcc", "posEntryMode", "merchent_zip3_country"]

    # Merge features
    df = merge_features(df, calc_time_features, time_cols)
    df = merge_features(df, calc_amount_features, amount_cols)
    df = merge_features(df, calc_account_features, account_cols)
    df = merge_features(df, calc_merchant_features, merchant_cols)

    # Hash categorical features (commented out as an example)
    # df = hash_features(df, categorical_features, n_features=20)

    # Save data if specified
    if save_data:
        df.to_csv("data/processed/features_data.csv", index=False)

    return df


def merge_features(df: pd.DataFrame, func, cols: list) -> pd.DataFrame:
    """Merge calculated features into DataFrame."""
    return df.merge(func(df[cols]), how="left", on="eventId")


def calc_time_features(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Extracts time-based features.

    Args:
        df_slice: DataFrame containing columns 'transactionTime', 'accountNumber' if needed.

    Returns:
        DataFrame with added 'transaction_hour', 'transaction_day', 'transaction_month',
        and 'time_since_last_transaction' columns.
    """
    # df_slice["transactionTime"] = pd.to_datetime(df_slice["transactionTime"])
    df_slice.loc[:, "transaction_hour"] = df_slice.loc[:, "transactionTime"].dt.hour
    df_slice.loc[:, "transaction_day"] = df_slice.loc[:, "transactionTime"].dt.day
    df_slice.loc[:, "transaction_month"] = df_slice.loc[:, "transactionTime"].dt.month
    df_slice = df_slice.sort_values(["accountNumber", "transactionTime"])
    df_slice["time_since_last_transaction"] = df_slice.groupby("accountNumber")[
        "transactionTime"
    ].diff()
    df_slice["time_since_last_transaction"] = (
        df_slice["time_since_last_transaction"].dt.total_seconds() / 60
    )
    df_slice["time_since_last_transaction"] = df_slice[
        "time_since_last_transaction"
    ].fillna(0)

    return df_slice.loc[
        :,
        [
            "eventId",
            "transaction_hour",
            "transaction_day",
            "transaction_month",
            "time_since_last_transaction",
        ],
    ]


def calc_amount_features(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Calculates amount-related features.

    Args:
        df_slice: DataFrame containing columns 'transactionAmount' and 'availableCash'.

    Returns:
        DataFrame with added 'amount_to_available_cash_ratio' and 'amount_binned' columns.
    """
    df_slice["amount_to_available_cash_ratio"] = (
        df_slice["transactionAmount"] / df_slice["availableCash"]
    )

    def apply_bins(amount):
        for label, upper_bound in AMOUNT_BIN_DICT.items():
            if amount <= upper_bound:
                return label

    df_slice["amount_binned"] = df_slice["transactionAmount"].apply(apply_bins)
    return df_slice.loc[
        :, ["eventId", "amount_to_available_cash_ratio", "amount_binned"]
    ]


def calc_account_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates account-based features.

    Args:
        df: The DataFrame containing transaction data.

    Returns:
        The DataFrame with 'transactions_per_account', 'average_transaction_amount_per_account'.
        amount_deviation_from_account_avg
    """

    df_account_features = df.groupby("accountNumber")[["transactionAmount"]].agg(
        ["count", "mean", "std"]
    )
    df_account_features.columns = [
        "transactions_per_account",
        "average_transaction_amount_per_account",
        "std_transaction_amount_per_account",
    ]

    # merge account features
    df = df.join(df_account_features, on="accountNumber")

    # transaction amount deviation from account avg
    account_avg_amount = (
        df[df.is_fraud == 0]
        .groupby("accountNumber")["transactionAmount"]
        .median()
        .to_dict()
    )
    df["amount_deviation_from_account_avg"] = df.apply(
        lambda row: row["transactionAmount"]
        - account_avg_amount.get(row["accountNumber"], row["transactionAmount"]),
        axis=1,
    )

    # fill missing values for std_transaction_amount_per_account with 0
    df["std_transaction_amount_per_account"] = df[
        "std_transaction_amount_per_account"
    ].fillna(0)

    return df.loc[
        :,
        [
            "eventId",
            "transactions_per_account",
            "average_transaction_amount_per_account",
            "std_transaction_amount_per_account",
            "amount_deviation_from_account_avg",
        ],
    ]


def calc_merchant_features(df_slice: pd.DataFrame) -> pd.DataFrame:
    """Calculates merchant-related features.

    Args:
        df_slice: DataFrame containing 'merchantId', 'accountNumber', 'merchantZip_imputed', 'merchantCountry'.

    Returns:
        DataFrame with added 'is_new_merchant', 'merchent_zip3_country' columns (and potentially others).
    """
    df_slice["is_new_merchant"] = ~df_slice.duplicated(
        subset=["accountNumber", "merchantId"], keep="first"
    )
    df_slice["merchent_zip3_country"] = (
        df_slice.merchantZip_imputed.str[:3] + df_slice.merchantCountry
    )
    return df_slice.loc[:, ["eventId", "is_new_merchant", "merchent_zip3_country"]]


def hash_features(
    df: pd.DataFrame, columns: List[str], n_features: int
) -> pd.DataFrame:
    """Applies feature hashing to categorical columns."""
    hasher = FeatureHasher(n_features=n_features, input_type="string")
    hashed_features = hasher.fit_transform(df[columns]).toarray()
    df_hashed = pd.DataFrame(
        hashed_features, columns=[f"hashed_{i}" for i in range(n_features)]
    )
    return pd.concat([df, df_hashed], axis=1)


# These features are pre-calculated on entire historical transaction data and store in the database for future use
def precalc_risk_scores(df: pd.DataFrame, risk_columns: Dict[str, str]) -> pd.DataFrame:
    """Calculates risk scores for multiple columns based on historical fraud data.

    Args:
        df: The DataFrame containing transaction data, including an 'is_fraud' column.
        risk_columns: A dictionary mapping the column names to their corresponding risk score names.
                      Example: {'merchantId': 'merchant_risk_score',
                                'merchantZip_imputed': 'zip_code_risk_score',
                                'merchantCountry': 'country_risk_score'}

    Returns:
        The DataFrame with added risk score columns.
    """

    for column, risk_score_name in risk_columns.items():
        if column in df.columns:
            fraud_events = (
                df[df["is_fraud"] == 1].groupby(column)["transactionAmount"].count()
            )
            total_events = df.groupby(column)["transactionAmount"].count()
            risk_scores = fraud_events / total_events
            risk_scores.index = risk_scores.index.astype(
                str
            )  # Ensure string indices for mapping

            df[risk_score_name] = df[column].map(risk_scores).fillna(0.0)

    logger.info(f"Calculated risk scores for columns: {list(risk_columns.values())}")

    return df


def create_auto_bins(X, y, feature_name, n_bins=10):
    """Creates bins for a feature based on target distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        feature_name (str): Name of the feature for binning.
        n_bins (int, optional): Number of bins to create. Defaults to 10.

    Returns:
        pd.Series: Series containing bin labels for each data point.
    """
    # Group by target value and calculate quantiles for transaction amounts
    fraud_quantiles = X.groupby(y)[feature_name].quantile([0.1, 0.2, ..., 0.9])

    # Assign bin labels based on quantiles
    def get_bin(x):
        for quantile, label in fraud_quantiles.iteritems():
            if x <= label:
                return label
        return label  # Ensures all values are assigned a bin

    X["bin_" + feature_name] = X[feature_name].apply(get_bin)
    return X["bin_" + feature_name]


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


def split_data(df, split_ratio=0.95):
    """Split the data into development and test data"""

    split_index = int(split_ratio * df.shape[0])
    return df.iloc[:split_index], df.iloc[split_index:]


if __name__ == "__main__":
    file_path = "data/processed/development_data.pkl"
    extract_features(pd.read_pickle(file_path))
