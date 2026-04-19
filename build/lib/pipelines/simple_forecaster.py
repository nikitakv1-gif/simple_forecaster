from utils.dataframe_utils import train_test_split
from core.models.utils import  seasonality
import pandas as pd
from core.models.ensemble import ensemble_calculate
from core.models.model_validation import multi_model_valid
from utils.plots_utils import plot_forecast_results
# from ..core.models import prophet

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
    y.index.freq = freq

    # определяем длинну теста, если входное число с плавающей точкой, то это доля, если целое, то это индекс
    y_train, y_test = train_test_split(y, test_size)

    #получаем сезонный цикл
    seasonal_periods = seasonal_periods if seasonal_periods != None else seasonality(freq)
    date_predict = pd.date_range(start=y_train.ds[0], periods=forecast_len, freq=freq)

    to_return = {}

    assert forecast_len < y_test or 0, "Убедитесь, что длинна прогноза больше тестовой части"   

    if 'prophet' in model_list:
        from ..core.models import prophet
        pred_stat['prophet'] = prophet.fit_predict(y_train, date_predict, freq).pipeline_info
     
    if 'arima' in model_list:
        from ..core.models import arima
        pred_stat['arima'] = arima.fit_predict(y_train, date_predict, seasonal_period).pipeline_info
        

    if 'es' in model_list:
        from ..core.models import es
        pred_stat['es'] = es.fit_predict(y_train, date_predict, seasonal_period).pipline_info

    if 'ensemble' in model_list:
        from ..core.models import ensemble_calculate
        pred_stat['ensemble'] = ensemble_calculate(pred_stat)

    to_return['pred_stat'] = pred_stat
    if y_test != None:
        to_return['test_res'] = multi_model_valid(pred_stat, y_test)

    if graph:
        to_return['plots'] = plot_forecast_results(pred_stat, y_train, y_test)
    
    return to_return