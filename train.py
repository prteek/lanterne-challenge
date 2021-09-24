import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer as FT
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform, randint, loguniform, beta
from sklearn.metrics import mean_squared_error, mean_poisson_deviance
import joblib
from sklearn.preprocessing import SplineTransformer

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from eda import add_hour_day_of_week
from sklearn.base import BaseEstimatorEstimator, TransformerMixin
from dabl import clean, explain

class ZIPRegressor(BaseEstimator, TransformerMixin):
    def __init__(self, zip_threshold=0, **kwargs):
        self.zip_threshold = zip_threshold
        self.model = PoissonRegressor(**kwargs)
    
    def fit(self, X,y):
        Xt = X[y>0]
        yt = y[y>0]
        self.model.fit(Xt,yt)
        return self
    
    def predict(self,X,y=None):
        prediction = (np.random.random() >= self.zip_threshold)*self.model.predict(X)
        return prediction
        
        

if __name__ == '__main__':
    
    
    df_train = pd.read_csv('train.csv', index_col=0)
    
    target = 'demand'
    
    X = df_train.query("demand>=0").drop(target, axis=1)
    y = df_train.query("demand>=0")[target]
    
    # Need to drop columns identified non informative in eda
    columns_to_drop = ['tourist_attractions', 'secondary_connectivity']
    drop_columns = ColumnTransformer([('drop_columns', 'drop', columns_to_drop)], remainder='passthrough') 
    
    # Can use PCA since many columns are integral and not all of them will be informative
    pca = PCA(n_components=0.95) # Arbitrarily pick threshold 0.95
    
    model = ZIPRegressor(zip_threshold=0.8, max_iter=1000) # We can treat the problem as Poisson regression since demand is an integer per 4 hour interval i.e. rate of an event
        
    pipeline = Pipeline([('time_preprocessing', FT(add_hour_day_of_week)),
                        ('drop_redundant_columns', drop_columns),
                         ('spline', SplineTransformer()),
                        ('pca', pca), 
                         ('scaler', StandardScaler()),
                        ('model', model)])
    
    
    rss = RandomizedSearchCV(pipeline, {'model__zip_threshold': beta(8,2), 'pca':[pca]}, 
                             n_iter=100, 
                             scoring=['neg_mean_squared_error'], 
                             refit='neg_mean_squared_error', 
                             cv=5, 
                             verbose=9,
                            n_jobs=-1)
    
    rss.fit(X,y)
    
    
    print('Mean squared error', -rss.best_score_)
        
    joblib.dump(rss, 'model.mdl')
          
    
    
    
        