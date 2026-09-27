from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


def calculate_adjusted_r2(r2, number_of_samples, number_of_features):
    return 1 - (
        (1 - r2)
        * (number_of_samples - 1)
        / (number_of_samples - number_of_features - 1)
    )


def calculate_regression_metrics(y_true, y_pred, number_of_features):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "adjusted r2": calculate_adjusted_r2(r2_score(y_true, y_pred), len(y_true), number_of_features)
    }
