import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL
import optuna
from src.core.models.model_validation import calculate_metrics
from src.utils.dataframe_utils import train_test_split


def stop_when_no_improvement(study, trial):

    window = 5

    if len(study.trials) < window:
        return

    best_recent = min(t.value for t in study.trials[-window:])
    best_global = study.best_value

    if best_recent >= best_global:
        study.stop()

def stl_arima_optimizer(y, 
                   seasonal_periods, 
                   arima_params = None,
                   trend = None,
                   n_trials=30):


    threshold = seasonal_periods * 2
    if len(y) >= threshold:
        y_train, y_val = train_test_split(y, 0.1)
    else:
        y_train = y
        y_val = None

    res = STL(y_train, period=seasonal_periods, robust=True).fit()

    y_deseason = y_train['y'].values - res.seasonal
    
    seasonal_pattern = res.seasonal[-seasonal_periods:]
    future_season = np.resize(seasonal_pattern, len(y_val))

    if arima_params == None:
        p, q, d = None, None, None 
    else:
        p, q, d = arima_params

    all_params = {'p' : p, 'd': d, 'q': q,
                  'm': seasonal_periods, 'trend': trend}
    def objective(trial):

        p = all_params['p'] if all_params['p'] != None else trial.suggest_int('p', 0, 3)
        d = all_params['d'] if all_params['d'] != None else trial.suggest_int('d', 0, 2)
        q = all_params['q'] if all_params['q'] != None else trial.suggest_int('q', 0, 3)
        trend = all_params['trend'] if all_params['trend'] != None else trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])

        if d >= 2 and trend != 'n':
            return 1e10
        elif d == 1 and trend not in ['n', 't']:
            return 1e10
        
        try:
            model = ARIMA(y_deseason, order=(p, d, q), trend=trend).fit()
            
            pred = model.forecast(len(y_val)) + future_season
            if y_val is not None:
                return calculate_metrics(y_val['y'].values, pred)['WAPE']
            else:
                return model.aic
        except Exception as e:
            print(e)
            return 1e10

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_params = study.best_params

    if y_val is not None:
        ansamble_weight = study.best_value
    else:
        ansamble_weight = 1

    for i in best_params:
        all_params[i] = best_params[i]
    order = (all_params['p'], all_params['d'], all_params['q'])
    best_trend = all_params['trend']

    return (order, best_trend, ansamble_weight)