import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.model_config import MODEL_CONFIGS, RANDOM_STATE

from sklearn.base import clone

NUMERICAL_COL = [
    "distance_km",
    "order_hour",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
]

CATEGORICAL_COL = [
    "Weatherconditions",
    "Type_of_vehicle",
    "Festival",
    "City",
    "Type_of_order",
]

ORDINAL_COL = ["Road_traffic_density"]


def create_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            ("onehot", OneHotEncoder(drop="if_binary",handle_unknown="ignore")),
        ]
    )

    ordinal_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", missing_values=pd.NA)),
            ("ordinal", OrdinalEncoder(
                categories=[["Low", "Medium", "High", "Jam"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERICAL_COL),
            ("categorical", categorical_pipeline, CATEGORICAL_COL),
            ("ordinal", ordinal_pipeline, ORDINAL_COL),
        ]
    )


def create_model_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", clone(model)),
        ]
    )


def create_pipelines():
    return {
        model_name: create_model_pipeline(model["estimator"])
        for model_name, model in MODEL_CONFIGS.items()
    }


def train_pipeline(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def train_pipelines(X_train, y_train):
    return {
        model_name: train_pipeline(pipeline, X_train, y_train)
        for model_name, pipeline in create_pipelines().items()
    }


def create_cross_validation():
    return KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )


def cross_validate_pipeline(pipeline, X_train, y_train, cv):
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        return_train_score=True
    )
    return {
        "validation_mae": -scores["test_mae"].mean(),
        "validation_mse": -scores["test_mse"].mean(),
        "validation_rmse": -scores["test_rmse"].mean(),
        "validation_r2": scores["test_r2"].mean(),
        "training_mae": -scores["train_mae"].mean(),
        "fit_time": scores["fit_time"].mean(),
    }


def cross_validate_pipelines(X_train, y_train):
    pipelines = create_pipelines()
    cv = create_cross_validation()
    results = {}

    for model_name, pipeline in pipelines.items():
        model_results = cross_validate_pipeline(
            pipeline,
            X_train,
            y_train,
            cv
        )
        results[model_name] = model_results
    return results
