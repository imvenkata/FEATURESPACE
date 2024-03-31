import pandas as pd
from src.config import POS_ENTRY_MODE_MAP

from typing import Dict


def load_data() -> pd.DataFrame:
    """Loads the transaction data, infers fraud labels, and prepares features.

    Returns:
        pandas.DataFrame: The processed transaction DataFrame.
    """

    # PLACEHOLDER: In real-world projects, load data from a database or cloud storage, update below lines with cloud
    # storage config and corresponding connection string and bucket name
    transaction_df = pd.read_csv("../data/transactions_obf.csv")
    labels = pd.read_csv("../data/labels_obf.csv")

    transaction_df = transaction_df.merge(labels, on="eventId", how="left")
    transaction_df["is_fraud"] = transaction_df["reportedTime"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )

    transaction_df = convert_data_types(
        transaction_df.copy()
    )  # Best practice to avoid in-place modification
    transaction_df = add_pos_entry_description(transaction_df)
    transaction_df.sort_values(
        "transactionTime", inplace=True
    )  # Explicit inplace modification
    transaction_df = fill_missing_zip_with_dict(transaction_df.copy())

    return transaction_df


def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Converts data types of columns for consistency.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with converted data types.
    """
    df["transactionTime"] = pd.to_datetime(df["transactionTime"], utc=True)
    df["eventId"] = df["eventId"].astype(str)
    df["accountNumber"] = df["accountNumber"].astype(str)
    df["merchantId"] = df["merchantId"].astype(str)
    df["mcc"] = df["mcc"].astype(str)
    df["merchantCountry"] = df["merchantCountry"].astype(str)
    df["merchantZip"] = df["merchantZip"].astype(str)
    df["posEntryMode"] = df["posEntryMode"].astype(int)
    df["transactionAmount"] = df["transactionAmount"].astype(float)
    df["availableCash"] = df["availableCash"].astype(float)
    return df


def add_pos_entry_description(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a description column based on the 'posEntryMode' code.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with the 'posEntryMode_description' column.
    """
    df["posEntryMode_description"] = df["posEntryMode"].map(POS_ENTRY_MODE_MAP)
    return df


def fill_missing_zip_with_dict(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing merchantZip.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with missing zip codes filled.
    """
    # Create a dictionary to store merchantId and most recent zip code
    merchant_zip_dict = {}
    df["merchantZip_imputed"] = df["merchantZip"]  # Copy for filled values

    for i in range(len(df)):
        current_merchant = df.loc[i, "merchantId"]
        current_zip = df.loc[i, "merchantZip"]

        # Update dictionary with current merchant and zip if not missing
        if not pd.isna(current_zip):
            merchant_zip_dict[current_merchant] = current_zip

        # Use the dictionary to fill missing zip codes (considering all past transactions)
        if pd.isna(df.loc[i, "merchantZip_imputed"]):
            if current_merchant in merchant_zip_dict:
                df.loc[i, "merchantZip_imputed"] = merchant_zip_dict[current_merchant]
            else:
                df.loc[i, "merchantZip_imputed"] = (
                    "missing_zip"  # No zip found for merchant
                )

    return df
