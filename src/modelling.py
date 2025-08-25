from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import sklearn
import numpy as np
import pandas as pd
from src.processing import process_routes


def regressor_pipeline(
        df:pd.DataFrame, 
        target_col:str, 
        model:sklearn.base.RegressorMixin, 
        test_size:float=0.2, 
        random_state:int=42
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply machine learning pipeline on training, the pipeline, splits the data, encode categorical variables
     and fit a regressor to training data and estimate prediction on test data

     Parameters:
     df: Dataframe of training data.
     target_col: name of dependant variable.
     model: scikit-learn classifier.
     test_size: proportion of data to keep for testing.
     random_state: random seed

     Returns:
     y_test: observed y values.
     predictions: predicted y values.
     model: trained model.
     """
    features = [col for col in df if col !=target_col]
    X = df[features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return y_test, X_test, predictions, model

def get_intertials(df):
    """
    find optimum number of clusters by visually inspecting elbow plot.

    Parameters:
    df: dataframe containing routes information.

    Returns
    inertias: errors per cluster.
    df_scaled: scaled dataframe of routes.
    scaler: fitted minmax scaler.
    """
    intertias = []
    df_coords = df[["routeCentroidLatitude", "routeCentroidLongitude"]]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_coords)
    K = range(1, 10)
    
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42).fit(df_scaled)
        intertias.append(kmeanModel.inertia_)
    return intertias, df_scaled, scaler

def get_predictions(
        df:pd.DataFrame, 
        df_scaled:pd.DataFrame
        ):
    """
    Get quantile prediction per class

    Parameters:
    df: dataframe containing routes information.
    df_scaled: scaled dataframe containing routes information.

    Returns:
    df_processed: processed routes dataframes with actual times and quantile ranges as predictions.
    t: dataframe containing the classes and thier qunatile ranges.
    model: trained kmeans model
    """
    cols_to_keep = ["firstToLastStopActualDurationHours", "lowerboundprediction", "upperboundprediction"]
    model = KMeans(n_clusters=3, random_state=42)
    classes = model.fit_predict(df_scaled)
    df["Class"] = classes
    df_processed, t = process_routes(df)
    return df_processed[cols_to_keep], t, model