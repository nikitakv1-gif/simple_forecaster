import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from src.core.optimizers.stl_arima_optimizer import stl_arima_optimizer
import pandas as pd

class STLArimaModel:

    def __init__(self, seasonal_periods = 12, arima_params = None, trend_params = None):
        self.is_fitted = False
        self.seasonal_periods = seasonal_periods
        self.arima_params = arima_params
        self.trend_params = trend_params

    def fit(self, y_train):

        if (self.arima_params is None or 
            self.trend_params is None):
            
            (self.arima_params, 
             self.trend_params,
             self.ensemble_weight) =  stl_arima_optimizer(y_train[['y']], 
                                                      self.seasonal_periods,
                                                      self.arima_params, 
                                                      self.trend_params)
            

        self.model = STLForecast(y_train['y'], ARIMA, model_kwargs=dict(order=self.arima_params, trend=self.trend_params, enforce_stationarity=False),
                                  period = self.seasonal_periods, robust=True).fit()
    
        train_preds = self.model.get_prediction(start=y_train.index[0], end=y_train.index[-1]).predicted_mean
        self.is_fitted = True
        self.fittedvalues = train_preds
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
    
                                                 
