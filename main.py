import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
from src.processing import process_legs, process_stops, generate_master_data, train_test_split
from src.plotting import plot_routes, plot_nulls, plot_correlation
from src.modelling import regressor_pipeline
from src.logging_config import logger

def main():
    df_routes = pd.read_csv("./data/input/training_routes.csv")
    df_legs = pd.read_csv("./data/input/training_legs.csv")
    df_stops = pd.read_csv("./data/input/training_stops.csv")
    logger.info(f"Reading raw data")
    df_legs_pro = process_legs(df=df_legs)
    df_stops_pro = process_stops(df=df_stops)
    df_master = generate_master_data(df_routes=df_routes, df_stops=df_stops_pro, df_legs=df_legs_pro)
    logger.info(f"Processed data generated sucessfully")
    XGBoostRegressor = GradientBoostingRegressor()
    y_test, predictions, model = regressor_pipeline(df=df_master, target_col="firstToLastStopActualDurationHours", model=XGBoostRegressor, test_size=0.2)
    logger.info(f"R squared of predictions is {r2_score(y_test, predictions)}")
    logger.info(f"Mean absolute error is {mean_absolute_error(y_test, predictions)}")


if __name__=="__main__":
    main()