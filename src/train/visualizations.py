import plotly.graph_objects as go


def fraud_transactions_pie_chart(
    total_transactions, fraudulent_transactions, fraud_amount, cost_multiplier=5
):
    """
    Creates a pie chart visualizing the proportion of fraudulent transactions
    and includes an annotation with estimated fraud cost.

    Args:
        total_transactions (int): Total number of transactions.
        fraudulent_transactions (int): Number of fraudulent transactions.
        fraud_amount (float): Total amount lost due to fraud.
        cost_multiplier (float, optional): Multiplier to estimate the total cost
                                          of fraud. Defaults to 5.
    """

    # Create the pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Legitimate Transactions", "Fraudulent Transactions"],
                values=[
                    total_transactions - fraudulent_transactions,
                    fraudulent_transactions,
                ],
                textinfo="label+value+percent",
                hole=0.34,
            )
        ]
    )  # hole parameter creates a donut chart

    # Add text annotation for the fraud amount
    fig.add_annotation(
        x=0.495,
        y=1,
        text=f"Total cost: £{fraud_amount * cost_multiplier:.2f}",
        font=dict(size=12),
        showarrow=True,
        arrowhead=3,
        ax=30,
        ay=-70,
    )

    # Show the plot
    fig.show()


def plot_performance_metrics(metrics, train_values, val_values, test_values):
    """
    Creates a grouped bar chart visualizing performance metrics across datasets.

    Args:
        metrics (list): List of metric names.
        train_values (list): List of metric values for the training set.
        val_values (list): List of metric values for the validation set.
        test_values (list): List of metric values for the test set.
    """

    # Create traces for each dataset
    trace_train = go.Bar(
        x=metrics,
        y=train_values,
        name="Train set",
        text=train_values,
        textposition="auto",
    )
    trace_val = go.Bar(
        x=metrics,
        y=val_values,
        name="Validation set",
        text=val_values,
        textposition="auto",
    )
    trace_test = go.Bar(
        x=metrics, y=test_values, name="Test set", text=test_values, textposition="auto"
    )

    # Configure layout
    layout = go.Layout(
        title="Performance Metrics",
        xaxis=dict(title="Metric"),
        yaxis=dict(title="Value"),
        barmode="group",
    )

    # Create the figure and plot the data
    fig = go.Figure(data=[trace_train, trace_val, trace_test], layout=layout)
    fig.show()


# Example Usage:
metrics = ["Precision", "Recall", "F1 Score", "ROC AUC"]
train_values = [0.63, 0.88, 0.77, 0.95]
val_values = [0.49, 0.80, 0.61, 0.90]
test_values = [0.30, 0.69, 0.42, 0.83]

plot_performance_metrics(metrics, train_values, val_values, test_values)

# Example Usage:
if __name__ == "__main__":
    fraud_transactions_pie_chart(117746, 875, 100773)

    metrics = ["Precision", "Recall", "F1 Score", "ROC AUC"]
    train_values = [0.63, 0.88, 0.77, 0.95]
    val_values = [0.49, 0.80, 0.61, 0.90]
    test_values = [0.30, 0.69, 0.42, 0.83]
    plot_performance_metrics(metrics, train_values, val_values, test_values)
