import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from dabl import clean, plot, detect_types
from dabl.plot import *
import sweetviz as sv


def add_hour_day_of_week(df):
    """Add hour and day of week from timestamp column and drop timestamp"""
    X = df.copy()
    X['timestamp'] = pd.to_datetime(X['timestamp'])
    X['hour'] = X['timestamp'].apply(lambda x: x.hour)
    X['day_of_week'] = X['timestamp'].apply(lambda x: x.day_of_week)
    X.drop('timestamp', axis=1, inplace=True)
    
    return X


if __name__ == "__main__":

    df_train = pd.read_csv("train.csv", index_col=0)
    
    report_full = sv.analyze(df_train, target_feat='demand')
    report_full.show_html(layout='vertical', filepath='report_full.html')
    
    """Observations from full report:
    1. There are over whelmingly high number of zeros in demand
    2. We need to look at the data in a comparative sense where only positive demand data is assessed against full data
    3. No missing values and data seems well formatted
    """
    
    comparative_report = sv.compare([df_train, 'full'], [df_train.query("demand>0"), 'positive_demand'], target_feat='demand')
    comparative_report.show_html(layout='vertical', filepath='report_comparative.html')
    
    """ Observations from comparative report of positive demand vs full range of demand values:
    1. The characteristics of positive demand values across variables look very similar to full range of demand values
    2. Because there are so many zeros in demand, they are wel distributed across range of values on other variables and do not introduce skewnes in data
    3. tourist_attraction and secondary_connectivity are non informative columns
    """
    
    df_clean = clean(df_train, target_col='demand', verbose=2)
    
    print(df_train.shape, df_clean.shape)
    
    """After cleaning the 2 non informative columns have been dropped"""
    
    