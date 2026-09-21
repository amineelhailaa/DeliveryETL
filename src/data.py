import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/dataset-1.csv"

CATEGORICAL_COLUMNS = ["Weather", "Traffic_Level", "Time_of_Day", "Vehicle_Type"]
NUMERICAL_COLUMNS = ["Order_ID", "Distance_km", "Preparation_Time_min", "Courier_Experience_yrs", "Delivery_Time_min"]


def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_delivery_data(df):
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()

    for column in CATEGORICAL_COLUMNS:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.strip()
            .str.title()
        )

    for column in NUMERICAL_COLUMNS:
        cleaned[column] = (
            pd.to_numeric(
                cleaned[column],
                errors="coerce"
            )
        )

    cleaned = cleaned.drop_duplicates().copy()
    cleaned = cleaned.dropna(subset=["Delivery_Time_min"]).copy()

    # cleaned["Delivery_Time_min"] = cleaned["Delivery_Time_min"].abs()
    # cleaned["Preparation_Time_min"] = cleaned["Preparation_Time_min"].abs()
    # cleaned["Distance_km"] = cleaned["Distance_km"].abs()
    # cleaned["Courier_Experience_yrs"] = cleaned["Courier_Experience_yrs"].abs()

    # cleaned["Weather"] = cleaned["Weather"].fillna(cleaned["Weather"].mode()[0])
    # cleaned["Traffic_Level"] = cleaned["Traffic_Level"].fillna(cleaned["Traffic_Level"].mode()[0])
    # cleaned["Time_of_Day"] = cleaned["Time_of_Day"].fillna(cleaned["Time_of_Day"].mode()[0])
    # cleaned["Courier_Experience_yrs"] = cleaned["Courier_Experience_yrs"].fillna(cleaned["Courier_Experience_yrs"].median())

    for column in NUMERICAL_COLUMNS:
        print(
            column,
            "min:", cleaned[column].min(),
            "max:", cleaned[column].max()
        )

    return cleaned.reset_index(drop=True)



def create_train_test_split(cleaned_df, test_size=0.2, random_state=42):
    #target
    X = cleaned_df.drop(columns=["Delivery_Time_min", "Order_ID", "Preparation_Time_min"])
    Y = cleaned_df["Delivery_Time_min"]
    x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return x_train, x_test , y_train, y_test
