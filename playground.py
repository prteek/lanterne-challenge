import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



if __name__ == '__main__':
    
    sns.set()
    
    df_train = pd.read_csv('train.csv', index_col=0)

    # Take a look at dataframe
    df_train.head()
    
    
    # Check if there may be missing values
    df_train.info()

    
    # Check the effect of weather
    weather_related_variables = ['tempC', 'precipMM']
    target = 'demand'
    df_weather = df_train[weather_related_variables+[target]]

    sns.pairplot(df_weather)
    
    # demand seems to be skewed so we can transform it to better visualize
    df_weather['log_demand'] = df_weather[target].apply(lambda x: np.log(x+1))
    
    sns.pairplot(df_weather[weather_related_variables + ['log_demand']])
    
    
    # Demand seems to be mostly zero (consequently log(demand+1) has zeros
    
    df_train[target].value_counts()
    # in addition demand is integer (could be the number of vehicles used by customers)
    
    # Let's remove zeros and have a look again
    
    sns.displot(df_weather.query("log_demand > 0"), x='tempC',y='log_demand', kind='kde')
    # Higher temperatures seem to slightly correlate with higher demand as expected 
    
    sns.displot(df_weather.query("log_demand > 0"), x='precipMM',y='log_demand', kind='kde')
    # In general higher precipitation can mean low demand
    
    

        
        
        
        
        