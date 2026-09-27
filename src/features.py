import pandas as pd
import numpy as np
from src.validation import validate_distance, validate_order_hour




def add_distance_km(df):

    lat1 = np.radians(df["Restaurant_latitude"])
    lon1 = np.radians(df["Restaurant_longitude"])
    lat2 = np.radians(df["Delivery_location_latitude"])
    lon2 = np.radians(df["Delivery_location_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2)
            * np.sin(dlon / 2) ** 2
    )

    df["distance_km"] = (
            2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    )
    df["distance_km"] = validate_distance(df["distance_km"])
    return df



def add_order_hour(df):
    df["order_hour"] = df["Time_Orderd"].dt.components["hours"]
    df["order_hour"] = validate_order_hour(df["order_hour"])

    return df



def create_features(df):
    df = add_distance_km(df)
    df = add_order_hour(df)
    return df
