from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor


RANDOM_STATE = 42


LINEAR_REGRESSION_SEARCH_SPACE = {
    "model__fit_intercept": [True, False],
}


RIDGE_SEARCH_SPACE = {
    "model__alpha": [
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ],
    "model__fit_intercept": [True, False],
}


DECISION_TREE_SEARCH_SPACE = {
    "model__max_depth": [
        5,
        10,
        15,
        20,
        None,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
        20,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        5,
        10,
    ],
    "model__max_features": [
        "sqrt",
        "log2",
        1.0,
    ],
}


RANDOM_FOREST_SEARCH_SPACE = {
    "model__n_estimators": [
        200,
        300,
        500,
    ],
    "model__max_depth": [
        8,
        12,
        16,
        24,
        None,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
        20,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        5,
        10,
    ],
    "model__max_features": [
        "sqrt",
        0.7,
        1.0,
    ],
}


MODEL_CONFIGS = {
    "dummy_baseline": {
        "estimator": DummyRegressor(
            strategy="median",
        ),
        "search_space": None,
        "n_iter": 0,
    },
    "linear_regression": {
        "estimator": LinearRegression(),
        "search_space": LINEAR_REGRESSION_SEARCH_SPACE,
        "n_iter": 2,
    },
    "ridge_regression": {
        "estimator": Ridge(),
        "search_space": RIDGE_SEARCH_SPACE,
        "n_iter": 10,
    },
    "decision_tree": {
        "estimator": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),
        "search_space": DECISION_TREE_SEARCH_SPACE,
        "n_iter": 12,
    },
    "random_forest": {
        "estimator": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "search_space": RANDOM_FOREST_SEARCH_SPACE,
        "n_iter": 15,
    },
}
