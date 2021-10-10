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
from eda import add_hour_day_of_week
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb


class ZIPRegressor(BaseEstimator, TransformerMixin):
    
    def __init__(self, clf, reg):
        self.clf = clf
        self.reg = reg
        
    def fit(self, X,y):            
        self.clf.fit(X, y>0)
        self.reg.fit(X[y>0], y[y>0])
        return self
    
    def predict(self, X,y=None):
        prediction = self.clf.predict(X)*self.reg.predict(X)
        return prediction
        
        

if __name__ == '__main__':
    
    
    df_train = pd.read_csv('train.csv', index_col=0)
    
    target = 'demand'
    
    X = df_train.query("demand>=0").drop(target, axis=1)
    y = df_train.query("demand>=0")[target]
    
    # Need to drop columns identified non informative in eda
    columns_to_drop = ['tourist_attractions', 'secondary_connectivity']
    drop_columns = ColumnTransformer([('drop_columns', 'drop', columns_to_drop)], remainder='passthrough') 
    
    hp = {
        'objective': 'reg:squarederror',
        'n_estimators': 200,
        'max_depth':5
    }
    
    model = model = ZIPRegressor(xgb.XGBClassifier(), xgb.XGBRegressor(**hp)) # We can treat the problem as Zero inflated

    pipeline = Pipeline([('time_preprocessing', FT(add_hour_day_of_week)),
                        ('drop_redundant_columns', drop_columns),
                        ('model', model)])
    
    pipeline.fit(X,y)
           
    joblib.dump(pipeline, 'model.mdl')

    
    
    
        