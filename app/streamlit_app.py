import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "delivery_eta_pipeline.joblib"
REPORT_DIR = PROJECT_ROOT / "reports"


@st.cache_resource
def load_pipeline(path, modified_ns):
    pipeline = joblib.load(path)
    model_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return pipeline, model_hash


@st.cache_data
def load_reports(directory, modified_times):
    directory = Path(directory)
    metrics = json.loads((directory / "final_metrics.json").read_text())
    training = pd.read_csv(directory / "training_eda.csv")
    predictions = pd.read_csv(directory / "test_predictions.csv")
    return metrics, training, predictions


st.set_page_config(page_title="Delivery ETA", page_icon="⏱", layout="wide")
st.title("Delivery ETA")
st.caption("Estimate delivery duration from the route, delivery conditions, and assigned courier.")

try:
    pipeline, model_hash = load_pipeline(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except Exception as error:
    st.error(f"Could not load the saved model: {error}")
    st.info("Save the fitted pipeline from the notebook to models/delivery_eta_pipeline.joblib.")
    st.stop()

preprocessor = pipeline.named_steps["preprocessor"]
category_columns = next(columns for name, _, columns in preprocessor.transformers_ if name == "categorical")
category_values = preprocessor.named_transformers_["categorical"].named_steps["onehot"].categories_
choices = dict(zip(category_columns, category_values))
traffic_choices = preprocessor.named_transformers_["ordinal"].named_steps["ordinal"].categories_[0]

predict_tab, data_tab, performance_tab = st.tabs(["Predict delivery", "Data", "Model performance"])

with predict_tab:
    st.subheader("Delivery details")
    st.write("Enter the conditions known for this delivery. All fields are required.")
    with st.form("delivery_form"):
        route, conditions = st.columns(2)
        with route:
            st.markdown("**Route and order**")
            restaurant_latitude = st.number_input("Restaurant latitude (degrees)", min_value=6.0, max_value=38.0, value=12.9716, format="%.6f")
            restaurant_longitude = st.number_input("Restaurant longitude (degrees)", min_value=68.0, max_value=98.0, value=77.5946, format="%.6f")
            delivery_latitude = st.number_input("Delivery latitude (degrees)", min_value=6.0, max_value=38.0, value=12.9916, format="%.6f")
            delivery_longitude = st.number_input("Delivery longitude (degrees)", min_value=68.0, max_value=98.0, value=77.6146, format="%.6f")
            st.caption("Coordinates are limited to the geographic region used in training. Distance is calculated by the model pipeline.")
            order_time = st.time_input("Order time", value=pd.Timestamp("12:00:00").time())
            order_type = st.selectbox("Order type", choices["Type_of_order"])
            city = st.selectbox("Area category", choices["City"], format_func=lambda value: "Metropolitan" if value == "Metropolitian" else value)
        with conditions:
            st.markdown("**Conditions and courier**")
            traffic = st.selectbox("Traffic", traffic_choices)
            weather = st.selectbox("Weather", choices["Weatherconditions"])
            vehicle_type = st.selectbox("Vehicle type", choices["Type_of_vehicle"], format_func=lambda value: value.replace("_", " "))
            festival = st.selectbox("Festival", choices["Festival"])
            age = st.number_input("Courier age (years)", min_value=1, value=25, step=1)
            rating = st.number_input("Courier rating (0–5)", min_value=0.0, max_value=5.0, value=4.5, step=0.1)
            vehicle_condition = st.selectbox("Vehicle condition code", [0, 1, 2, 3], index=1, help="Original dataset codes; their descriptive meanings have not been verified.")
            multiple_deliveries = st.number_input("Multiple deliveries (count)", min_value=0, value=1, step=1)
        submitted = st.form_submit_button("Estimate delivery time", type="primary")

    if submitted:
        required_values = [restaurant_latitude, restaurant_longitude, delivery_latitude, delivery_longitude, age, rating, multiple_deliveries]
        if order_time is None or any(value is None or not np.isfinite(value) for value in required_values):
            st.error("Please provide a valid order time and finite numbers for every numeric field.")
        elif not (6 <= restaurant_latitude <= 38 and 6 <= delivery_latitude <= 38 and 68 <= restaurant_longitude <= 98 and 68 <= delivery_longitude <= 98):
            st.error("Coordinates must be within the region shown in the form.")
        elif age < 1 or age % 1 != 0 or not 0 <= rating <= 5 or multiple_deliveries < 0 or multiple_deliveries % 1 != 0:
            st.error("Use a positive whole-number age, a rating from 0 to 5, and a nonnegative whole-number delivery count.")
        else:
            input_df = pd.DataFrame([{
                "Restaurant_latitude": restaurant_latitude,
                "Restaurant_longitude": restaurant_longitude,
                "Delivery_location_latitude": delivery_latitude,
                "Delivery_location_longitude": delivery_longitude,
                "Time_Orderd": pd.Timedelta(hours=order_time.hour, minutes=order_time.minute, seconds=order_time.second),
                "Delivery_person_Age": age,
                "Delivery_person_Ratings": rating,
                "Vehicle_condition": vehicle_condition,
                "multiple_deliveries": multiple_deliveries,
                "Weatherconditions": weather,
                "Road_traffic_density": traffic,
                "Type_of_vehicle": vehicle_type,
                "Festival": festival,
                "City": city,
                "Type_of_order": order_type,
            }])
            try:
                prediction = float(pipeline.predict(input_df)[0])
                if not np.isfinite(prediction) or prediction < 0:
                    raise ValueError("The model returned an invalid delivery duration.")
                st.success(f"Estimated delivery time: {prediction:.1f} minutes")
                st.caption("An estimate based on historical deliveries; actual delivery time may differ.")
            except Exception as error:
                st.error(f"Could not estimate this delivery: {error}")

report_paths = [REPORT_DIR / name for name in ["final_metrics.json", "training_eda.csv", "test_predictions.csv"]]
report_error = None
try:
    report_times = tuple(path.stat().st_mtime_ns for path in report_paths)
    metrics, training_eda, test_predictions = load_reports(str(REPORT_DIR), report_times)
    if metrics["model_sha256"] != model_hash:
        raise ValueError("The reports belong to a different saved model.")
    if not {"Time_taken(min)", "distance_km", "Road_traffic_density", "Weatherconditions"}.issubset(training_eda.columns):
        raise ValueError("The training EDA report is missing required columns.")
    if not {"actual_value", "predicted_value"}.issubset(test_predictions.columns):
        raise ValueError("The test prediction report is missing required columns.")
    if training_eda.empty or test_predictions.empty:
        raise ValueError("The saved reports contain no observations.")
    if any(not isinstance(metrics[key], (int, float)) or not np.isfinite(metrics[key]) for key in ["mae", "rmse", "r2"]):
        raise ValueError("The saved performance metrics are invalid.")
except Exception as error:
    report_error = f"Reports are missing or outdated: {error}. Run the notebook's Export app reports cell after evaluation."

with data_tab:
    st.subheader("Training data")
    if report_error:
        st.info(report_error)
    else:
        st.caption(f"{len(training_eda):,} training deliveries. Test observations are excluded from these charts.")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.histplot(data=training_eda, x="Time_taken(min)", bins=30, ax=axes[0])
        axes[0].set(xlabel="Delivery time (minutes)", ylabel="Deliveries", title="Delivery duration")
        axes[1].scatter(training_eda["distance_km"], training_eda["Time_taken(min)"], alpha=0.08, s=8)
        axes[1].set(xlabel="Straight-line distance (km)", ylabel="Delivery time (minutes)", title="Distance and delivery time")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.boxplot(data=training_eda, x="Road_traffic_density", y="Time_taken(min)", order=list(traffic_choices), ax=axes[0])
        axes[0].set(xlabel="Traffic", ylabel="Delivery time (minutes)", title="Delivery time by traffic")
        sns.boxplot(data=training_eda, x="Weatherconditions", y="Time_taken(min)", ax=axes[1])
        axes[1].set(xlabel="Weather", ylabel="Delivery time (minutes)", title="Delivery time by weather")
        axes[1].tick_params(axis="x", rotation=30)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with performance_tab:
    st.subheader("Test-set performance")
    if report_error:
        st.info(report_error)
    else:
        mae_col, rmse_col, r2_col = st.columns(3)
        mae_col.metric("MAE (minutes)", f"{metrics['mae']:.2f}")
        rmse_col.metric("RMSE (minutes)", f"{metrics['rmse']:.2f}")
        r2_col.metric("R²", f"{metrics['r2']:.3f}")
        st.write(f"On {len(test_predictions):,} test deliveries, predictions differ from observed times by {metrics['mae']:.2f} minutes on average. This is not a guaranteed error bound.")
        actual = test_predictions["actual_value"]
        predicted = test_predictions["predicted_value"]
        residuals = actual - predicted
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].scatter(actual, predicted, alpha=0.15, s=10)
        lower, upper = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        axes[0].plot([lower, upper], [lower, upper], "r--", label="Perfect prediction")
        axes[0].set(xlabel="Actual time (minutes)", ylabel="Predicted time (minutes)", title="Actual versus predicted")
        axes[0].legend()
        sns.histplot(residuals, bins=30, ax=axes[1])
        axes[1].axvline(0, color="red", linestyle="--")
        axes[1].set(xlabel="Actual − predicted time (minutes)", ylabel="Deliveries", title="Prediction errors")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Model: {metrics['model_name']}. Evaluation uses the existing 80/20 split with random seed 42.")
    st.info("The source data contains inconsistent ages for the same courier IDs. Treat this educational model's results as provisional until those records are verified.")
