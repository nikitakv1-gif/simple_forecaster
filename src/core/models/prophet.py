from prophet import Prophet
import pandas as pd
from src.core.optimizers.prophet_optimizer import prophet_optimizer

class ProphetModel:

    def __init__(self, seasonal_periods = 12, freq = 'MS', **kwargs):
        self.model = Prophet(**kwargs)
        self.is_fitted = False
        self.freq = freq
        self.seasonal_periods = seasonal_periods

    def fit(self, y_train):

        (self.ensemble_weight) = prophet_optimizer(y_train, self.seasonal_periods, self.freq)

        self.model.fit(y_train)
        self.is_fitted = True
        self.fittedvalues = pd.Series(self.model.predict(y_train[['ds']])['yhat'].values, index = y_train.ds)
        self.train_len = len(y_train)
        return self
        
    def predict(self, predict_date):
        
        assert self.is_fitted == True, 'model is not fitted'

        p_test = self.model.predict(predict_date)['yhat'].values
        
        self.predvalues = pd.Series(p_test, predict_date.ds)

        self.pipeline_info = {'model': self.model, 'train_pred': self.fittedvalues, 'test_pred': self.predvalues, 'ensemble_weight': self.ensemble_weight} 
        
        return self.predvalues

    def fit_predict(self, y_train, predict_date):
        
        self.fit(y_train)
        self.predict(predict_date)

        return self.predvalues
    
                                                 
