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
from dabl import clean, explain
import pymc3 as pm


class ZIPRegressor(BaseEstimator, TransformerMixin):
    
    def fit(self, X,y):
        with pm.Model() as ZIP_reg:
            phi = pm.Beta('phi', 1,1)
            alpha = pm.Normal('alpha', 0,10)
            beta = pm.Normal('beta',0,10,shape=X.shape[1])
            theta = pm.math.exp(alpha+X@beta)
            y1=pm.ZeroInflatedPoisson('y1', phi,theta,observed=y)
            self.trace_zip_reg=pm.sample(1000)

        return self
    
    def predict(self,X,y=None):
        prediction = np.mean(np.exp(self.trace_zip_reg['alpha'].T + Xt@self.trace_zip_reg['beta'].T), axis=1)
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
    
    model = ZIPRegressor() # We can treat the problem as Poisson regression since demand is an integer per 4 hour interval i.e. rate of an event

    pipeline = Pipeline([('time_preprocessing', FT(add_hour_day_of_week)),
                        ('drop_redundant_columns', drop_columns),
                        ('pca', pca), 
                         ('scaler', StandardScaler()),
                        ('model', model)])
    
    pipeline.fit(X,y)
           
    joblib.dump(pipeline, 'model.mdl')

    
    
    
        