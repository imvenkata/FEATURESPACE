from typing import Tuple
import pandas as pd


def cost_and_savings_analysis(
    data: pd.DataFrame,
    fraud_cost_multiplier: float = 5,
) -> tuple[float, float, float, float, float]:
    """
    Estimates the potential savings, costs, and other metrics using a fraud detection model.

    Args:
        data: A DataFrame containing 'is_fraud', 'predicted_label', and 'transactionAmount' columns.
        fraud_cost_multiplier: Multiplier for the cost of fraudulent transactions.
    Returns:
        A tuple containing:
            estimated_savings,
            false_negative_cost,
            total_transaction_amount,
            fraud_cost_without_model,
    """

    # Check if expected columns are present
    if not set(["is_fraud", "predicted_label", "transactionAmount"]).issubset(
        data.columns
    ):
        raise ValueError("Missing required columns in the DataFrame")

    # Calculate the total transaction amount
    total_transaction_amount = data["transactionAmount"].sum()

    # Calculate the cost of fraudulent transactions without the model
    fraud_cost_without_model = (
        data["is_fraud"] * data["transactionAmount"] * fraud_cost_multiplier
    ).sum()

    # Calculate the cost of fraudulent transactions with the model
    false_negative_cost = (
        (data["is_fraud"] & ~data["predicted_label"])
        * data["transactionAmount"]
        * fraud_cost_multiplier
    ).sum()

    # Calculate the savings
    estimated_savings = fraud_cost_without_model - false_negative_cost

    return (
        estimated_savings,
        false_negative_cost,
        total_transaction_amount,
        fraud_cost_without_model,
    )
