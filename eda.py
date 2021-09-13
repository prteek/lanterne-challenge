import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import SplineTransformer
from hulearn.experimental import parallel_coordinates
import altair as alt



if __name__ == '__main__':
    
    sns.set()
    
    df_train = pd.read_csv('train.csv', index_col=0)

    # Take a look at dataframe
    df_train.head()
    
    
    # Check if there may be missing values
    df_train.info()

    # ------------------------------------------------------- #
    # Check the effect of weather
    weather_related_variables = ['tempC', 'precipMM']
    target = 'demand'
    df_weather = df_train[weather_related_variables+[target]]

    sns.pairplot(df_weather)
    
    # demand seems to be skewed so we can transform it to better visualize
    df_weather['log_demand'] = df_weather[target].apply(lambda x: np.log(x+1))
    
    sns.pairplot(df_weather.drop(target, axis=1))
    
    
    # Demand seems to be mostly zero (consequently log(demand+1) has zeros
    
    df_train[target].value_counts()
    # in addition demand is integer (could be the number of vehicles used by customers)
    
    # Let's remove zeros and have a look again
    
    sns.displot(df_weather.query("log_demand > 0"), x='tempC',y='log_demand', kind='kde')
    # Higher temperatures seem to slightly correlate with higher demand as expected 
    
    sns.displot(df_weather.query("log_demand > 0"), x='precipMM',y='log_demand', kind='kde')
    # In general higher precipitation can mean low demand
    
    
    # ------------------------------------------------------- #
    # Check if connectivity affects demand
    connectivity_related_variables  = ['cycleway_connectivity', 'tertiary_connectivity', 'secondary_connectivity', 'station_count']
    
    df_connectivity = df_train[connectivity_related_variables + [target]]
    df_connectivity['log_demand'] = df_connectivity[target].apply(lambda x: np.log(x+1))
    
    sns.pairplot(df_connectivity.drop(target, axis=1))
    
    # Secondary connectivity is zero so perhaps we can drop it from analysis
    # Others seem to be influencing the demand in a way that higher connectivity leads to lower demand    
    
    # ------------------------------------------------------- #
    # Can time be an influencing factor
    
    df_train['timestamp'] = pd.to_datetime(df_train['timestamp'])
    
    df_train['hour'] = df_train['timestamp'].apply(lambda x: x.hour)
    
    df_train.plot.scatter(x='hour', y=target)
    # Early hours have less demand
    
    df_train.plot.scatter(x='hour', y='travelling_proportion')
    # Evening hours also correspond to when higher proportion of people travel
    
    df_train.plot.scatter(x='travelling_proportion', y=target)

    
    # day of week
    df_train['day_of_week'] = df_train['timestamp'].apply(lambda x: x.day_of_week)
    
    df_train.plot.scatter(x='day_of_week', y=target)
    # not much significant impact of day of week. It was expected if weekdays or weekends couls be significantly difference buut that isn't the case
    
    datetime_df = df_train[['hour','day_of_week', target]].groupby(['hour','day_of_week']).agg(sum).reset_index()
    
    datetime_df = datetime_df.pivot('hour','day_of_week', target).fillna(0)
    
    sns.heatmap(datetime_df)
    # Higher demands happen around afternoon and evenings towards end of week
    
    # ------------------------------------------------------- #
    # Analysing remainder (lack of time)
    
    other_columns = ['count_commercial' , 'count_retail', 'cover_commercial' ,'cover_retail', 'tourist_attractions', 'node_id']
    df_others = df_train[other_columns + [target]]
    df_others['log_demand'] = df_others[target].apply(lambda x: np.log(x+1))
    sns.pairplot(df_others.drop(target, axis=1).query("log_demand>0"))
    
    # there are no tourist attractions so this column can be dropped
    # generally higher cover and counts relate with declining demand
    
    
    # ------------------------------------------------------- #
    # Analysing distance (lack of time)
    
    distance_columns = ['distance_to_recreation_ground' , 'distance_to_park', 'distance_to_retail' ,'distance_to_commercial', 'node_id']
    df_distance = df_train[distance_columns + [target]]
    df_distance['log_demand'] = df_distance[target].apply(lambda x: np.log(x+1))
    sns.pairplot(df_distance.drop(target, axis=1).query("log_demand>0"))
    
    
        
        
        