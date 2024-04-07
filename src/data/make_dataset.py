import pandas as pd
from src.config import POS_ENTRY_MODE_MAP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_data_preparation(data_dir: str, save_data=False) -> pd.DataFrame:
    """Loads transaction data, infers fraud labels, and prepares features.

    Args:
        data_dir: Directory containing transaction and label data.

    Returns:
        pandas.DataFrame: The processed transaction DataFrame.
    """
    transaction_df = pd.read_csv(f"{data_dir}/raw/transactions_obf.csv")
    labels = pd.read_csv(f"{data_dir}/raw/labels_obf.csv")

    transaction_df = transaction_df.merge(labels, on="eventId", how="left")
    transaction_df["is_fraud"] = transaction_df["reportedTime"].apply(
        lambda x: 0 if pd.isnull(x) else 1
    )

    transaction_df = preprocess_dataframe(transaction_df)

    # Save the processed data
    if save_data:
        transaction_df.to_pickle(f"{data_dir}/processed/transaction_df.pkl")

    return transaction_df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the DataFrame, converting data types, adding features, and handling missing values.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    df = remove_duplicate_transactions(df.copy())
    df = convert_data_types(df.copy())
    df = add_pos_entry_description(df)
    df.sort_values("transactionTime", inplace=True)
    df = impute_missing_merchant_zip(df.copy())
    return df


def remove_duplicate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate transactions from the DataFrame.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with duplicates removed.
    """
    df = df.drop_duplicates()
    return df


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


def impute_missing_merchant_zip(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing merchantZip with the most recently known zip.

    Args:
        df: The input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with missing zip codes filled.
    """
    merchant_zip_dict = {}
    df["merchantZip_imputed"] = df["merchantZip"]  # Copy for filled values

    for i in range(len(df)):
        current_merchant = df.loc[i, "merchantId"]
        current_zip = df.loc[i, "merchantZip"]

        if not pd.isna(current_zip):
            merchant_zip_dict[current_merchant] = current_zip

        if pd.isna(df.loc[i, "merchantZip_imputed"]):
            if current_merchant in merchant_zip_dict:
                df.loc[i, "merchantZip_imputed"] = merchant_zip_dict[current_merchant]
            else:
                df.loc[i, "merchantZip_imputed"] = "missing_zip"

    return df


if __name__ == "__main__":
    data_dir = "/Users/venkata.medabala/Projects/ml/featurespace/data"
    run_data_preparation(data_dir)
    logging.info("Data preparation completed. Processed data saved.")
