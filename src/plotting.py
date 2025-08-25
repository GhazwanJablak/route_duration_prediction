import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_routes(
        df:pd.DataFrame
        ):
    """
    Function to plot route coordinates in the UK.

    Parameters:
    df: input dataframe with coordinates.
    """
    fig = px.scatter_mapbox(
    df,
    lat="routeCentroidLatitude",
    lon="routeCentroidLongitude",
    color="firstToLastStopActualDurationHours",
    hover_name="id",
    title="routes",
    zoom=8,
    center=dict(lat=54.5, lon=-3.5),  
    mapbox_style="carto-positron" 
    )
    fig.show()


def plot_nulls(
        df:pd.DataFrame
        ):
    """
    Function to plor null values in dataframe

    Parameters:
    df: input dataframe to be inspected for nulls.
    """
    null_vals = (df.isnull().sum()/df.shape[0])*100
    _, ax = plt.subplots()
    ax.bar(null_vals.index, null_vals.values)
    ax.bar_label(ax.containers[0])
    ax.set_ylabel("Null percentage")
    ax.tick_params(axis="x", rotation=75)
    ax.set_title("Null values per feature in raw dataset")
    plt.show()


def plot_correlation(
        df: pd.DataFrame
        ):
    """
    Function to plot correlation matrix of numeric columns in dataframe

    Parameters:
    df: input dataframe to be inspected for correlation.
    """
    num_col = [col for col in df.columns if df[col].dtypes!="object"]
    df[num_col].corr()
    _, ax = plt.subplots()
    sns.heatmap(df[num_col].corr(), annot=True, ax = ax)
    plt.show()