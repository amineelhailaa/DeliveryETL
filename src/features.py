import pandas as pd
import numpy as np




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
    return df



def add_order_hour(df):
    df["order_hour"] = df["Time_Orderd"].dt.total_seconds()/3600

    return df



def create_features(df):
    df = add_distance_km(df)
    df = add_order_hour(df)
    return df
