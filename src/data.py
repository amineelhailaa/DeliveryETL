import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
RAW_DATA_PATH = Path(__file__).resolve().parents[1] / "data/raw/dataset-2.csv"



CATEGORICAL_COLUMNS = ["Weatherconditions",
                       "Road_traffic_density",
                       "Type_of_vehicle",
                       ]

NUMERICAL_COLUMNS = [
                     "Restaurant_latitude",
                     "Restaurant_longitude",
                     "Delivery_location_latitude",
                     "Delivery_location_longitude",
                     "Time_taken(min)",
                    ]



TIME_COLUMNS = [
    "Time_Order_picked"
]





def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def clean_delivery_data(df):
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=["ID",
                                    "Delivery_person_ID",
                                    "Delivery_person_Age",
                                    "Delivery_person_Ratings",
                                    "Type_of_order",
                                    "multiple_deliveries",
                                    "Order_Date",
                                    "Festival",
                                    "City",
                                    "Vehicle_condition"])


    cleaned.columns = cleaned.columns.str.strip()




    #conditons prob in weather
    cleaned["Weatherconditions"] = (
        cleaned["Weatherconditions"].astype("string")
        .str.strip()
        .str.removeprefix("conditions")
        .str.strip()
        .replace(["NaN", "nan", "None", "null", ""], pd.NA)
        .str.title()
    )

    #nan in weather
    weather_mode = cleaned["Weatherconditions"].mode()
    cleaned["Weatherconditions"] = cleaned["Weatherconditions"].fillna(weather_mode.iloc[0])


    for column in CATEGORICAL_COLUMNS:
        #categorical features:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.strip()
            .replace(["NaN", "nan", "None", "null", ""], pd.NA)
            .str.title()
        )







    #Time features:
    for column in TIME_COLUMNS:
        cleaned[column] = (
            cleaned[column].astype("string")
            .str.strip()
            .replace(["NaN", "nan", "None", "null", ""], pd.NA)
        )
        cleaned[column] = pd.to_timedelta(cleaned[column], errors="coerce")





    #time taken min problem
    cleaned["Time_taken(min)"]= pd.to_numeric(
        cleaned["Time_taken(min)"].str.removeprefix("(min)"), errors="coerce"
    )

    for column in NUMERICAL_COLUMNS:
        cleaned[column] = (
            pd.to_numeric(
                cleaned[column],
                errors="coerce"
            )
        )


    #fix latitude longtitude problem
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

    cleaned.loc[latitude_wrong | longitude_wrong , columns] =(
        cleaned.loc[latitude_wrong | longitude_wrong, columns].abs()
    )






    #Road traffic density

    cleaned["Road_traffic_density"] = cleaned["Road_traffic_density"].replace({
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Jam": 3
    }).copy()

    cleaned["Road_traffic_density"] =  cleaned["Road_traffic_density"].fillna(cleaned["Road_traffic_density"].mode()[0])



    # Keep coordinates located within India
    valid_coordinates = (
            cleaned["Restaurant_latitude"].between(6, 38)
            & cleaned["Restaurant_longitude"].between(68, 98)
            & cleaned["Delivery_location_latitude"].between(6, 38)
            & cleaned["Delivery_location_longitude"].between(68, 98)
    )

    cleaned = cleaned.loc[valid_coordinates].copy()



    cleaned = cleaned.dropna(subset=[
        "Time_taken(min)",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Time_Order_picked"
    ]).copy()





    # cleaned["Delivery_Time_min"] = cleaned["Delivery_Time_min"].abs()
    # cleaned["Preparation_Time_min"] = cleaned["Preparation_Time_min"].abs()
    # cleaned["Distance_km"] = cleaned["Distance_km"].abs()
    # cleaned["Courier_Experience_yrs"] = cleaned["Courier_Experience_yrs"].abs()

    # cleaned["Weather"] = cleaned["Weather"].fillna(cleaned["Weather"].mode()[0])
    # cleaned["Traffic_Level"] = cleaned["Traffic_Level"].fillna(cleaned["Traffic_Level"].mode()[0])
    # cleaned["Time_of_Day"] = cleaned["Time_of_Day"].fillna(cleaned["Time_of_Day"].mode()[0])
    # cleaned["Courier_Experience_yrs"] = cleaned["Courier_Experience_yrs"].fillna(cleaned["Courier_Experience_yrs"].median())

    return cleaned.reset_index(drop=True)




def create_train_test_split(cleaned_df, test_size=0.2, random_state=42):
    #target
    X = cleaned_df.drop(columns=["Delivery_Time_min", "Order_ID", "Preparation_Time_min"])
    Y = cleaned_df["Delivery_Time_min"]
    x_train, x_test , y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return x_train, x_test , y_train, y_test
