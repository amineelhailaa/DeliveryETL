import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

NUMERICAL_COL  = [
    "distance_km",
    "Road_traffic_density",
    "order_hour",
]

CATEGORICAL_COL = [
    "Weatherconditions",
    "Type_of_vehicle"
]

MODELS = {
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge( alpha = 1.0),
    "decision_tree": DecisionTreeRegressor( random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}


def create_preprocessor():
    numeric_pipline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipelines = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        [
            ("numeric", numeric_pipline, NUMERICAL_COL),
            ("categorical", categorical_pipelines, CATEGORICAL_COL)
        ]
    )




def create_model_pipeline(model):
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])


def create_pipelines():
    return {model_name: create_model_pipeline(model) for model_name, model in MODELS.items()}




def train_pipeline(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def train_piplines(piplines, X_train, y_train):
    return {model_name: train_pipeline(pipeline, X_train, y_train) for model_name, pipeline in piplines.items()}
