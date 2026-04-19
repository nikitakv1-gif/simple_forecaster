import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from src.core.optimizers.arima_optimizer import arima_optimizer
import pandas as pd

class ArimaModel:

    def __init__(self, seasonal_periods = 12, arima_params = None, seasonal_params = None, trend_params = None):
        self.is_fitted = False
        self.seasonal_periods = seasonal_periods
        self.arima_params = arima_params
        self.seasonal_params = seasonal_params
        self.trend_params = trend_params

    def fit(self, y_train):
        if (self.arima_params is None or 
            self.seasonal_params is None or 
            self.trend_params is None):
            
            (self.arima_params, 
             self.seasonal_params, 
             self.trend_params,
             self.ensemble_weight) =  arima_optimizer(y_train[['y']], 
                                                      self.seasonal_periods,
                                                      self.arima_params, 
                                                      self.seasonal_params, 
                                                      self.trend_params)
        
        self.model = ARIMA(y_train[['y']], order = self.arima_params,
                           seasonal_order = self.seasonal_params,
                           trend = self.trend_params).fit()
        

        self.is_fitted = True
        self.fittedvalues = pd.Series(self.model.predict().values, index = y_train.ds)
        self.train_len = len(y_train[['y']])
        
        return self
        
    def predict(self, predict_date):
        
        assert self.is_fitted == True, 'model is not fitted'

        forecast_p = self.model.forecast(len(predict_date))
        self.predvalues = pd.Series(forecast_p.values, predict_date.ds)

        self.pipeline_info = {'model': self.model, 
                              'train_pred': self.fittedvalues, 
                              'test_pred': self.predvalues,
                              'ensemble_weight': self.ensemble_weight} 
        
        return self.predvalues

    def fit_predict(self, y_train, predict_date):
        
        self.fit(y_train)
        self.predict(predict_date)

        return self.predvalues
    
                                                 
