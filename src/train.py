from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


NUMERICAL_COL = [
    "distance_km",
    "Road_traffic_density",
    "order_hour",
]

CATEGORICAL_COL = [
    "Weatherconditions",
    "Type_of_vehicle",
]

MODELS = {
    "dummy_baseline": DummyRegressor(strategy="median"),
    "linear_regression": LinearRegression(),
    "ridge_regression": Ridge(alpha=1.0),
    "decision_tree": DecisionTreeRegressor(random_state=42),
    "random_forest": RandomForestRegressor(
          n_estimators=300,
          max_depth=16,
          min_samples_split=10,
          min_samples_leaf=5,
          max_features=0.7,
          random_state=42,
          n_jobs=-1,
      )
}


def create_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERICAL_COL),
            ("categorical", categorical_pipeline, CATEGORICAL_COL),
        ]
    )


def create_model_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", model),
        ]
    )


def create_pipelines():
    return {
        model_name: create_model_pipeline(model)
        for model_name, model in MODELS.items()
    }


def train_pipeline(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def train_pipelines( X_train, y_train):
    return {
        model_name: train_pipeline(pipeline, X_train, y_train)
        for model_name, pipeline in create_pipelines().items()
    }



def create_cross_validation():
    return KFold(
        n_splits = 5,
        shuffle = True,
        random_state=42
    )


def cross_validate_pipeline(pipeline ,X_train, y_train, cv):
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


    for model_name , pipeline in pipelines.items():
        model_results = cross_validate_pipeline(
            pipeline,
            X_train,
            y_train,
            cv
        )
        results[model_name] = model_results
    return results


