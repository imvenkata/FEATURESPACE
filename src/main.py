from src.config import (
    EVENT_ID,
    FEATURES,
    LABEL,
    DATA_DIR,
    CATEGORICAL_FEATURES,
    CATBOOST_PARAM_GRID,
)

from src.train.model_utils import (
    prepare_data_for_modeling,
    split_into_xy,
    perform_stratified_split,
    apply_smote,
    build_catboost_model,
    perform_grid_search,
    save_model,
    evaluate_model,
)

if __name__ == "__main__":
    # Load data
    development_df, test_df = prepare_data_for_modeling(
        data_dir=DATA_DIR, event_id=EVENT_ID, features=FEATURES, label=LABEL
    )

    # Split data X,y
    X, y = split_into_xy(development_df)

    # Split data into train and validation
    X_train, X_val, y_train, y_val = perform_stratified_split(X, y)

    # set eventId as index
    X_train = X_train.set_index("eventId")
    X_val = X_val.set_index("eventId")

    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote(
        X_train, y_train, CATEGORICAL_FEATURES
    )

    # Build CatBoost model
    model = build_catboost_model()

    # Perform Grid Search and select best model
    best_model = perform_grid_search(
        model,
        param_grid=CATBOOST_PARAM_GRID,
        X=X_train_balanced,
        y=y_train_balanced,
        cat_features=CATEGORICAL_FEATURES,
    )

    # Save the model for later use
    save_model(best_model, "../models/catboost_model.cbm")

    # Evaluate the model

    # Predict on the train, validation and test data

    evaluate_model(best_model, X_train, y_train)
    evaluate_model(best_model, X_val, y_val)
    evaluate_model(best_model, test_df, test_df["is_fraud"])
