import logging
from typing import Tuple, List

import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

from src.data.make_dataset import run_data_preparation
from src.features.build_features import (
    extract_features,
    precalc_risk_scores,
)

from src.config import TEST_DATA_RATIO
from joblib import dump, load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data_for_modeling(
    data_dir: str, event_id: List[str], features: List[str], label: List[str]
) -> Tuple["pd.DataFrame", "pd.DataFrame"]:
    """Prepares data for modeling, including data splitting, feature extraction, and risk score calculation.

    TODO: exclude test data from feature engineering process to prevent data leakages

    Args:
        data_dir: Directory containing the data files.
        event_id: List of columns containing event identifiers.
        features: List of feature columns.
        label: List of label columns.

    Returns:
        A tuple containing the development and test DataFrames.
    """
    logger.info("Load data for modeling")
    transaction_df = run_data_preparation(data_dir=data_dir)

    logger.info("Extracting features")
    features_df = extract_features(transaction_df)

    # scale the features
    scaled_features_data = scale_features(features_df, feature_for_scaling)

    # create development and test data by splitting the data by time, 95% of the data is used for training
    # and 5% is used for testing. THe test data should not be part of feature engineering or model training.
    dev_features_data, test_features_data = split_data(scaled_features_data)

    dev_features_data = precalc_risk_scores(
        df=dev_features_data, risk_columns=risk_columns
    )

    # Merge test features data with merchant risk score based on merchantId
    test_features_data = test_features_data.merge(
        dev_features_data[["merchantId", "merchant_risk_score"]].drop_duplicates(),
        on="merchantId",  # Join on merchantId
        how="left",  # Use left join to retain all rows from test_features_data
    )

    # Merge with zip code risk score based on merchantZip_imputed
    test_features_data = test_features_data.merge(
        dev_features_data[
            ["merchantZip_imputed", "zip_code_risk_score"]
        ].drop_duplicates(),
        on="merchantZip_imputed",  # Join on merchantZip_imputed
        how="left",  # Use left join to retain all rows from test_features_data
    )

    # Merge with country risk score based on merchantCountry
    test_features_data = test_features_data.merge(
        dev_features_data[["merchantCountry", "country_risk_score"]].drop_duplicates(),
        on="merchantCountry",  # Join on merchantCountry
        how="left",  # Use left join to retain all rows from test_features_data
    )

    # fill missing merchant_risk_score, zip_code_risk_score, country_risk_score with 0
    test_features_data[
        ["merchant_risk_score", "zip_code_risk_score", "country_risk_score"]
    ] = test_features_data[
        ["merchant_risk_score", "zip_code_risk_score", "country_risk_score"]
    ].fillna(
        0
    )

    selected_columns = list(set(event_id + features + label))
    dev_features_data = dev_features_data[selected_columns]
    test_features_data = test_features_data[selected_columns]

    return dev_features_data, test_features_data


# Risk scores for merchantId, merchantZip_imputed, and merchantCountry
risk_columns = {
    "merchantId": "merchant_risk_score",
    "merchantZip_imputed": "zip_code_risk_score",
    "merchantCountry": "country_risk_score",
}

feature_for_scaling = [
    "availableCash",
    "transactionAmount",
    "time_since_last_transaction",
    "amount_deviation_from_account_avg",
    "transactions_per_account",
    "average_transaction_amount_per_account",
    "std_transaction_amount_per_account",
]


def split_data(
    df: pd.DataFrame,
    split_ratio: float = TEST_DATA_RATIO,
):
    """Split the data into development and test data.
    Randomly select samples within the last 30%
    """

    # split_index = int(split_ratio * df.shape[0])
    # df.iloc[:split_index], df.iloc[split_index:]

    # Calculate the length of the data
    total_length = len(df)

    # Calculate the start index for the last 30% of the data
    start_index_last_30 = int(0.3 * total_length)

    # Select the last 30% of the data
    last_30_percent = df.iloc[start_index_last_30:]

    # Randomly sample 5% of the total data frame from the last 30% as test_df
    test_sample_size = int(TEST_DATA_RATIO * total_length)
    test_df = last_30_percent.sample(n=test_sample_size, random_state=42)

    # Select the remaining data (excluding test_df) as dev_df
    dev_df = df.drop(test_df.index)

    return dev_df, test_df


def scale_features(df, feature_for_scaling, scaling_method="minmax"):
    """Applies scaling to specified features of a DataFrame.

    Args:
        df: The DataFrame to be scaled.
        feature_for_scaling: A list of column names to scale.
        scaling_method: The scaling method to use. Can be either "minmax" or "standard".

    Returns:
        The DataFrame with scaled features.
    """
    logger.info(f"Scaling features using {scaling_method} method")

    if scaling_method == "minmax":
        scaler = MinMaxScaler()
    elif scaling_method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling_method. Use 'minmax' or 'standard'")

    df[feature_for_scaling] = scaler.fit_transform(df[feature_for_scaling])
    return df


def split_into_xy(df: "pd.DataFrame") -> Tuple["pd.DataFrame", "pd.Series"]:
    """Splits a DataFrame into features (X) and labels (y).

    Args:
        df: The DataFrame to split.

    Returns:
        A tuple containing the features (X) and labels (y).
    """
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    return X, y


def perform_stratified_split(
    X: "pd.DataFrame", y: "pd.Series"
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.Series", "pd.Series"]:
    """Performs a stratified split on the input data.

    Args:
        X: The features DataFrame.
        y: The labels Series.

    Returns:
        A tuple containing the train and validation features (X_train, X_val) and labels (y_train, y_val).
    """
    logger.info("Performing stratified split")
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.10, random_state=42)
    for train_index, val_index in sss.split(X, y):
        return (
            X.iloc[train_index],
            X.iloc[val_index],
            y.iloc[train_index],
            y.iloc[val_index],
        )


def apply_smote(X_train, y_train, categorical_features=None):
    """Applies SMOTE to balance the training data."""

    logger.info("Applying SMOTE")
    smote_enn = SMOTENC(
        sampling_strategy="minority",
        categorical_features=categorical_features,
        random_state=42,
    )

    X_train_balanced, y_train_sample_balanced = smote_enn.fit_resample(X_train, y_train)

    return X_train_balanced, y_train_sample_balanced


def build_catboost_model():
    """Builds a CatBoost model."""

    logger.info("Building CatBoost model")
    cat_model = CatBoostClassifier(random_seed=42, verbose=False)
    return cat_model


def perform_grid_search(model, param_grid, X, y, cat_features=None):
    """Performs a grid search to find the best hyperparameters for the model."""

    if cat_features:
        categorical_features_indices = [
            index for index, name in enumerate(X.columns) if name in cat_features
        ]

    logger.info("Performing Grid Search")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=False,
    )

    # Fit the Grid Search
    grid_search.fit(X, y, cat_features=categorical_features_indices)

    # Best Model and Parameters
    best_model = grid_search.best_estimator_

    return best_model


def save_model(model, filename):
    """Saves the CatBoost model to a file."""
    logger.info(f"Model saved to {filename}")
    dump(model, f"{filename}.joblib")


# load model
def load_model(filename):
    """Loads a fraud model from a file."""

    model = load(f"{filename}.joblib")

    return model


def evaluate_model(model, X, y_true):
    """Calculates and prints evaluation metrics for a given model."""

    logger.info("Model Performance")
    y_pred = model.predict(X)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_true, y_pred))
