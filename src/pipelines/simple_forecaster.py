from src.utils.dataframe_utils import train_test_split
from src.core.models.utils import  seasonality
import pandas as pd
from src.core.models.ensemble import ensemble_calculate
from src.core.models.model_validation import multi_model_valid


def multi_model_forecast(
        y, 
        forecast_len, 
        test_size = None, 
        seasonal_periods = None, 
        freq = 'MS',
        model_list = [],
        graph = False):

    """
    Функция для прогнозирования на основе нескольких "простых" моделей
    """
    pred_stat = {}
                       
    y.columns = ['ds', 'y']
    
    y['ds'] = pd.to_datetime(y['ds'])
    
    y.index = y['ds'] 
    
    y.index.freq = freq

    # определяем длинну теста, если входное число с плавающей точкой, то это доля, если целое, то это индекс
    y_train, y_test = train_test_split(y, test_size)

    #получаем сезонный цикл
    seasonal_periods = seasonal_periods if seasonal_periods != None else seasonality(freq)
    next_date = y_train.ds.iloc[-1] + pd.tseries.frequencies.to_offset(freq)
    date_predict = pd.DataFrame(pd.date_range(start=next_date, periods=forecast_len, freq=freq), columns = ['ds'])

    to_return = {}

    if test_size is not None and test_size != 0:
        assert forecast_len > len(y_test['y']), "Убедитесь, что длинна прогноза больше тестовой части"   

    if 'prophet' in model_list:
        from src.core.models.prophet import ProphetModel
        prophet = ProphetModel(freq=freq, seasonal_periods = seasonal_periods)
        prophet.fit_predict(y_train, date_predict)
        pred_stat['prophet'] = prophet.pipeline_info
     
    if 'stl_arima' in model_list:
        from src.core.models.stl_arima import STLArimaModel
        arima = STLArimaModel(seasonal_periods=seasonal_periods)
        arima.fit_predict(y_train, date_predict)
        pred_stat['arima'] = arima.pipeline_info
        
    print('es')
    
    if 'es' in model_list:
        from src.core.models.es import ESModel
        es = ESModel(seasonal_periods=seasonal_periods)
        es.fit_predict(y_train, date_predict)
        pred_stat['es'] = es.pipeline_info

    if 'ensemble' in model_list:
        from src.core.models.ensemble import ensemble_calculate
        pred_stat['ensemble'] = ensemble_calculate(pred_stat, y_train, date_predict)

    to_return['pred_stat'] = pred_stat
    if test_size is not None and test_size != 0:
        to_return['test_res'] = multi_model_valid(pred_stat, y_test['y'].values)

    if graph:
        from src.utils.plots_utils import plot_forecast_results
        to_return['plots'] = plot_forecast_results(pred_stat, y_train, y_test)
    
    return to_return