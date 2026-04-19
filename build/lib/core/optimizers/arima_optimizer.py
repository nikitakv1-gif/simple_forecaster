import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import optuna
from ..models.model_validation import calculate_metrics
from ...utils.dataframe_utils import train_test_split

def arima_optimizer(y, 
                    seasonal_periods,
                    arima_params = (None, None, None), 
                    seasonal_params = (None, None, None), 
                    trend_params = None, 
                    n_trials=30):
    
    p, d, q = arima_params
    P, D, Q = seasonal_params
    m = seasonal_period
    trend = trend_params

    all_params = {'p' : p, 'd': d, 'q':q,
                  'P' : P, 'D': D, 'Q':Q,
                  'm': m, 'trend':trend}
    
    threshold = seasonal_periods * 3
    if len(y) >= threshold:
        y_train, y_val = train_test_split(y, 0.2)
    else:
        y_train = y
        y_val = None

    def objective(trial):
        p = p if p != None else trial.suggest_int('p', 0, 6)
        d = d if d != None else trial.suggest_int('d', 0, 3)
        q = q if q != None else trial.suggest_int('q', 0, 6)

        P = P if P != None else trial.suggest_int('P', 0, 6)
        D = D if D != None else trial.suggest_int('D', 0, 3)
        Q = Q if Q != None else trial.suggest_int('Q', 0, 6)

        trend = trend if trend else trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
        
        try:
            model = ARIMA(y_train, order=(p, d, q), seasonal_order = (P, D, Q, m), trend = trend).fit()

            if y_val != None:
                return calculate_metrics(model.forecast(len(y_val)), y_val)['WAPE']
            else:
                return model.aic
        except:
            return 1e10

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params

    if y_val != None:
        ansamble_weight = study.best_value
    else:
        ansamble_weight = 1

    for i in best_params:
        all_params[i] = best_params[i]
    order = (all_params['p'], all_params['d'], all_params['q'])
    seasonal_order = (all_params['P'], all_params['D'], all_params['Q'], m)
    best_trend = all_params['trend']

    return (order, seasonal_order, best_trend, ansamble_weight)