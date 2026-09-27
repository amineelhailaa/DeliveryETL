import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.validation import (
    validate_age,
    validate_ratings,
    validate_vehicle_condition,
    validate_multiple_deliveries,
    validate_weather,
    validate_vehicle_type,
    validate_order_type,
    validate_festival,
    validate_city,
    validate_traffic,
)

RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/dataset-2.csv"

CATEGORICAL_COLUMNS = ["Weatherconditions",
                       "Road_traffic_density",
                       "Type_of_vehicle",
                       "Type_of_order",
                       "Festival",
                       "City"
                       ]

NUMERICAL_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Time_taken(min)",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries"
]

TIME_COLUMNS = [
    "Time_Order_picked",
    "Time_Orderd",
]


def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_delivery_data(df):
    cleaned = df.copy()
    # cleaned = cleaned.drop(columns=["ID",
    #                                 "Delivery_person_ID",
    #                                 "Delivery_person_Ratings",
    #                                 "Type_of_order",
    #                                 "multiple_deliveries",
    #                                 "Festival",
    #                                 "Order_Date",
    #                                 "City",
    #                                 "Vehicle_condition"])

    cleaned.columns = cleaned.columns.str.strip()

    # conditons prob in weather
    cleaned["Weatherconditions"] = (
        cleaned["Weatherconditions"].astype("string")
        .str.strip()
        .str.removeprefix("conditions")
        .str.strip()
        .replace(["NaN", "nan", "None", "null", ""], pd.NA)
        .str.title()
    )

    # nan in weather
    # weather_mode = cleaned["Weatherconditions"].mode()
    # cleaned["Weatherconditions"] = cleaned["Weatherconditions"].fillna(weather_mode.iloc[0])

    for column in CATEGORICAL_COLUMNS:
        # categorical features:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.strip()
            .replace(["NaN", "nan", "None", "null", ""], pd.NA)
            .str.title()
        )

    # Time features:
    for column in TIME_COLUMNS:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.strip()
            .replace(["NaN", "nan", "None", "null", ""], pd.NA)
        )
        cleaned[column] = pd.to_timedelta(cleaned[column], errors="coerce")

    # time taken min problem
    cleaned["Time_taken(min)"] = pd.to_numeric(
        cleaned["Time_taken(min)"].str.removeprefix("(min)"), errors="coerce"
    )

    for column in NUMERICAL_COLUMNS:
        cleaned[column] = (
            pd.to_numeric(
                cleaned[column],
                errors="coerce"
            )
        )

    # fix latitude longtitude problem
    latitude_wrong = (
            cleaned["Restaurant_latitude"] * cleaned["Delivery_location_latitude"] < 0
    )
    longitude_wrong = (
            cleaned["Restaurant_longitude"] * cleaned["Delivery_location_longitude"] < 0
    )

    columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
    ]

    cleaned.loc[latitude_wrong | longitude_wrong, columns] = (
        cleaned.loc[latitude_wrong | longitude_wrong, columns].abs()
    )

    # Road traffic density

    # cleaned["Road_traffic_density"] = cleaned["Road_traffic_density"].replace({
    #     "Low": 0,
    #     "Medium": 1,
    #     "High": 2,
    #     "Jam": 3
    # }).copy()

    # cleaned["Road_traffic_density"] = cleaned["Road_traffic_density"].fillna(cleaned["Road_traffic_density"].mode()[0])

    # Keep coordinates located within India
    valid_coordinates = (
            cleaned["Restaurant_latitude"].between(6, 38)
            & cleaned["Restaurant_longitude"].between(68, 98)
            & cleaned["Delivery_location_latitude"].between(6, 38)
            & cleaned["Delivery_location_longitude"].between(68, 98)
    )

    cleaned = cleaned.loc[valid_coordinates].copy()


    ########################### VALIDATION ###############################
    cleaned["Delivery_person_Age"] = validate_age(cleaned["Delivery_person_Age"])
    cleaned["Delivery_person_Ratings"] = validate_ratings(cleaned["Delivery_person_Ratings"])
    cleaned["Vehicle_condition"] = validate_vehicle_condition(cleaned["Vehicle_condition"])
    cleaned["multiple_deliveries"] = validate_multiple_deliveries(cleaned["multiple_deliveries"])
    cleaned["Weatherconditions"] = validate_weather(cleaned["Weatherconditions"])
    cleaned["Type_of_vehicle"] = validate_vehicle_type(cleaned["Type_of_vehicle"])
    cleaned["Type_of_order"] = validate_order_type(cleaned["Type_of_order"])
    cleaned["Festival"] = validate_festival(cleaned["Festival"])
    cleaned["City"] = validate_city(cleaned["City"])
    cleaned["Road_traffic_density"] = validate_traffic(cleaned["Road_traffic_density"])













    cleaned = cleaned.dropna(subset=[
        "Time_taken(min)",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Time_Order_picked"
    ]).copy()

    # hundling time
    # estimate_order_time = (cleaned["Time_Order_picked"] - pd.Timedelta(minutes=10)) % pd.Timedelta(days=1)
    # cleaned["Time_Orderd"] = cleaned["Time_Orderd"].fillna(estimate_order_time)



    return cleaned.reset_index(drop=True)


def create_train_test_split(cleaned_df, test_size=0.2, random_state=42):
    # target
    X = cleaned_df.drop(columns=["Time_taken(min)"])
    Y = cleaned_df["Time_taken(min)"]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
