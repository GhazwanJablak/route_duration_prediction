from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import sklearn
import numpy as np
import pandas as pd


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
    return y_test, predictions, model