from statsmodels.tsa.holtwinters import ExponentialSmoothing
from ..optimizers import es_optimizer
import pandas as pd

class ESModel:

    def __init__(self, seasonal_periods = None, 
                 trend = None, seasonal = None, 
                 damped_trend = None, seasonal_periods = None):
        
        self.is_fitted = False
        self.seasonal_periods = seasonal_period
        self.parameters = None
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend
        self.seasonal_periods = seasonal_periods

    def fit(self, y_train):
        if (self.parameters is None):
            (self.trend, 
            self.seasonal, 
            self.damped_trend, 
            self.seasonal_periods, 
            self.ansamble_weight) = es_optimizer(y_train, 
                                                trend = self.trend,
                                                damped_trend = self.damped_trend,
                                                seasonal = self.seasonal, 
                                                seasonal_periods = self.seasonal_periods).best_params

        self.model = ExponentialSmoothing(endog = y_train, 
                                          seasonal_periods = self.seasonal_periods,
                                          trend = self.trend, damped_trend = self.damped_trend,
                                          seasonal = self.seasonal, seasonal_periods = self.seasonal_periods).fit()
        
        self.is_fitted = True

        self.fittedvalues = pd.Series(self.model.fittedvalues.values, index = y_train.ds)
        self.train_len = len(y_train)
        
        return self
        
    def predict(self, predict_date):
        
        assert self.is_fitted == True, 'model is not fitted'

        forecast_p = self.model.forecast(len(predict_date))
        self.predvalues = pd.Series(forecast_p, predict_date.ds)

        self.pipeline_info = {'model': self.model, 
                              'train_pred': self.fittedvalues, 
                              'test_pred': self.predvalues,
                              'ensemble_weight': self.ansamble_weight} 
        return self.predvalues

    def fit_predict(self, y_train, predict_date):
        
        self.fit(y_train)
        self.predict(predict_date)

        return self.predvalues
    
                                                 
