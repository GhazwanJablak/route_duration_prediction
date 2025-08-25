import pandas as pd
import numpy as np
from geopy.distance import distance
from functools import reduce


def generate_master_data(
        df_routes:pd.DataFrame, 
        df_legs:pd.DataFrame,
        df_stops: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Merge multiple dataset to generate master dataset for training a model.

    Parameters:
    df_routes: dataframe containing route information.
    df_legs: dataframe containing legs information.
    df_stops: dataframe containing stops information.

    Returns:
    df_master: combined dataset.
    """
    df_list = [df_routes, df_legs, df_stops]
    cols_to_drop = ["routeId", "routeDate", "routeCentroidLatitude", "routeCentroidLongitude"]
    df_routes.rename(columns={"id":"routeId"}, inplace=True)
    df_master = reduce(lambda  left,right: pd.merge(left,right,on=['routeId'], how='inner'), df_list)
    df_master["averageSpeed"] = df_master["DrivingDistance"]/df_master["firstToLastStopEstimatedTravelTimeHours"]
    return df_master.drop(cols_to_drop, axis=1)

def train_test_split(
        df:pd.DataFrame, 
        test_size:float=0.2
        ) -> pd.DataFrame:
    """
    Customized train test split function.

    Parameters:
    df: training dataframe.
    test_size: percentage representing test data.

    Returns:
    df_train: training dataset.
    df_test: testing dataset.
    """
    to_drop_cols = ["routeId", "routeDate"]
    test_split = int(np.round(df["routeId"].nunique()*test_size))
    test_routes = df["routeId"].unique().tolist()[1-test_split:]
    df_train = df.loc[~df["routeId"].isin(test_routes),:]
    df_test = df.loc[df["routeId"].isin(test_routes),:]
    return df_train.drop(to_drop_cols, axis=1), df_test.drop(to_drop_cols, axis=1)

def process_legs(
        df:pd.DataFrame
        ) -> pd.DataFrame:
    """
    Calculating driving distance per route covering all legs.

    Parameters:
    df: dataframe containing legs information

    Returns:
    grouper: legs data aggregated to route id level. 
    """
    df["DrivingDistance"] = df.sort_values(by="routeId").apply(lambda x: distance((x["fromLatitude"], x["fromLongitude"]), (x["toLatitude"],x["toLongitude"])).km, axis = 1)
    grouper = df.groupby("routeId").agg({"travelTimeSeconds":"sum", "DrivingDistance":'sum'}).reset_index()
    return grouper.drop("travelTimeSeconds", axis=1)



def process_stops(
        df:pd.DataFrame
        ) -> pd.DataFrame:
    """
    Fetching first and last stops coordinates for all routes.

    Parameters:
    df: dataframe containing stops information

    Returns:
    grouper: stops data aggregated to route id level. 
    """
    grouper = df.groupby("routeId").agg({
        "stopLatitude": ["first", "last"],
        'stopLongitude': ["first", "last"]
    }).reset_index()
    
    grouper.columns = ["routeId", "start_latitude", "end_latitude", "start_longitude", "end_longitude"]
    return grouper
