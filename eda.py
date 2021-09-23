import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from dabl import clean, plot, detect_types, EasyPreprocessor, SimpleRegressor
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
    4. We can keep the data along with zeros for any further analysis since removing them does not change the nature of relationships with demand significantly
    3. tourist_attraction and secondary_connectivity are non informative columns
    """
    
    df_clean = clean(df_train, target_col='demand', verbose=2)
    
    print(df_train.shape, df_clean.shape)
    
    """After cleaning the 2 non informative columns have been dropped"""
 
    # Add hour of day and day of week
    df_clean = add_hour_day_of_week(df_clean)
    
    chart = (alt
             .Chart(df_clean
                    .groupby(['hour', 'day_of_week'])
                    .agg({'demand':'mean'})
                    .reset_index()
                   )
             .mark_rect()
             .encode(x='hour:O', 
                     y='day_of_week:O', 
                     color='demand:Q', 
                     tooltip=['hour:Q', 'day_of_week:Q', 'demand:Q'])
             .interactive()
            )
    
    chart
    
    """Observations from hour of day and day of week
    1. Weekends appear busier (Saturday [5] and Sunday [6])
    2. Afternoon and evening hours are busier than mornings
    """
    
    # Some exploratory modelling 
    ep = EasyPreprocessor()
    X = ep.fit_transform(df_clean.drop('demand', axis=1))
    y = df_clean['demand'].values

    reg = SimpleRegressor()
    
    reg.fit(X,y)

    """R2 is so poor that if the model was a human it'd be Jon Snow. 
    And that is despite the zeros included in the data"""
    
    