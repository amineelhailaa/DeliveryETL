from sklearn import pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.model_config import *
from src.train import create_model_pipeline, create_cross_validation


def tune_model(model_name, X_train, y_train):
    config = MODEL_CONFIGS[model_name]
    estimator = config["estimator"]
    search_space = config["search_space"]
    n_iter = config["n_iter"]

    if search_space is None:
        raise ValueError( f"{model_name} does not have a tuning serach space")

    pipeline = create_model_pipeline(estimator)
    tuning_cv = create_cross_validation()

    search = RandomizedSearchCV(
        estimator = pipeline,
        param_distributions= search_space,
        n_iter = n_iter,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        cv = tuning_cv,
        random_state= RANDOM_STATE,
        refit= "mae",
        return_train_score= True,
        verbose= 1
    )

    search.fit(X_train, y_train)
    return search