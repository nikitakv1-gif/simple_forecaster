import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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


def arima_optimizer(y, 
                    seasonal_periods,
                    arima_params = None, 
                    seasonal_params = None, 
                    trend_params = None, 
                    n_trials=30):
    
    if arima_params == None:
        p, d, q = None, None, None
    else:
        p, d, q = arima_params

    if seasonal_params == None:
        P, D, Q = None, None, None
    else:
        P, D, Q = seasonal_params

    m = seasonal_periods

    all_params = {'p' : p, 'd': d, 'q':q,
                  'P' : P, 'D': D, 'Q':Q,
                  'm': m, 'trend': trend_params}
    
    threshold = seasonal_periods * 3
    if len(y) >= threshold:
        y_train, y_val = train_test_split(y, 0.1)
    else:
        y_train = y
        y_val = None

    if all_params['trend'] != None:
        if 'c' in all_params['trend']:
            diff_space_min = 0
            diff_space_max = 0
        else:
            diff_space_min = 0
            diff_space_max = 2
            diff_space_max_D = 1
    else:
        diff_space_min = 0
        diff_space_max = 2
        diff_space_max_D = 1

    def objective(trial):
        p = all_params['p'] if all_params['p'] != None else trial.suggest_int('p', 0, 6)
        P = all_params['P'] if all_params['P'] != None else trial.suggest_int('P', 0, 6)
        q = all_params['q'] if all_params['q'] != None else trial.suggest_int('q', 0, 2)
        Q = all_params['Q'] if all_params['Q'] != None else trial.suggest_int('Q', 0, 2)

        d = all_params['d'] if all_params['d'] != None else trial.suggest_int('d', diff_space_min, diff_space_max)
        D = all_params['D'] if all_params['D'] != None else trial.suggest_int('D', diff_space_min, diff_space_max_D)

        trend = all_params['trend'] if all_params['trend'] else trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])

        if d + D >= 2 and trend != 'n':
            return 1e10
        elif d + D == 1 and trend not in ['n', 't']:
            return 1e10

        try:
            model = ARIMA(y_train, order=(p, d, q), seasonal_order = (P, D, Q, m), trend = trend).fit()

            if y_val is not None:
                return calculate_metrics(y_val['y'].values, model.forecast(len(y_val)))['WAPE']
            else:
                return model.aic
        except Exception as e:
            print(e)
            return 1e10

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_params = study.best_params

    if len(y_val) > 0:
        ansamble_weight = study.best_value
    else:
        ansamble_weight = 1

    for i in best_params:
        all_params[i] = best_params[i]
    order = (all_params['p'], all_params['d'], all_params['q'])
    seasonal_order = (all_params['P'], all_params['D'], all_params['Q'], m)
    best_trend = all_params['trend']

    return (order, seasonal_order, best_trend, ansamble_weight)


