from prophet import Prophet
import pandas as pd
from ..models.utils import calculate_metrics

class ProphetModel:

    def __init__(self, freq = 'MS', **kwargs):
        self.model = Prophet(**kwargs)
        self.is_fitted = False
        self.freq = freq

    def fit(self, y_train):
        self.model.fit(y_train)
        self.is_fitted = True
        self.fittedvalues = pd.Series(self.model.predict(y_train['ds'])['yhat'].values, index = y_train.ds)
        self.train_len = len(y_train)
        return self
        
    def predict(self, predict_date):
        
        assert self.is_fitted == True, 'model is not fitted'

        forecast_p = self.model.predict(predict_date)
        
        p_test = forecast_p.iloc[self.train_len:]['yhat'].values
        self.predvalues = pd.Series(p_test, predict_date.ds)

        self.pipeline_info = {'model': self.model, 'train_pred': self.fittedvalues, 'test_pred': self.predvalues} 
        
        return self.predvalues

    def fit_predict(self, y_train, predict_date):
        
        self.fit(y_train)
        self.predict(predict_date)

        return self.predvalues
    
                                                 
