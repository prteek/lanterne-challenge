import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
import joblib
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor



def add_hour_day_of_week(df):
    """Add hour and day of week from timestamp column and drop timestamp"""
    X = df.copy()
    X['timestamp'] = pd.to_datetime(X['timestamp'])
    X['hour'] = X['timestamp'].apply(lambda x: x.hour)
    X['day_of_week'] = X['timestamp'].apply(lambda x: x.day_of_week)
    X.drop('timestamp', axis=1, inplace=True)
    
    return X



if __name__ == '__main__':
    
    
    df_train = pd.read_csv('train.csv', index_col=0)
    
    target = 'demand'
    
    X = df_train.query("demand>0").drop(target, axis=1)
    y = df_train.query("demand>0")[target]
    # Need to drop columns identified non informative in eda
    columns_to_drop = ['tourist_attractions', 'secondary_connectivity']
    drop_columns = ColumnTransformer([('drop_columns', 'drop', columns_to_drop)], remainder='passthrough') 
    
    # Can use PCA since many columns are integral and not all of them will be informative
    pca = PCA(n_components=0.95, whiten=True) # Arbitrarily pick threshold 0.95
    
    model = PoissonRegressor(max_iter=200) # We can treat the problem as Poisson regression since demand is an integer per 4 hour interval i.e. rate of an event
    
    model = Lasso()
    
    pipeline = Pipeline([('time_preprocessing', FT(add_hour_day_of_week)),
                        ('drop_redundant_columns', drop_columns),
                        ('pca', pca),
                        ('splie_transform', SplineTransformer()),
                        ('model', model)])
    
    
    rss = RandomizedSearchCV(pipeline, {'model__alpha': loguniform(0.001,10), 'pca':[pca, None]}, 
                             n_iter=100, 
                             scoring=['neg_mean_squared_error', 'neg_mean_poisson_deviance'], 
                             refit='neg_mean_squared_error', 
                             cv=5, 
                             verbose=9,
                            n_jobs=-1)
    
    rss.fit(X,y)
    
    
    print('Mean squared error', -cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error'))
        
    joblib.dump(pipeline, 'model.mdl')
          
    
    
    
        