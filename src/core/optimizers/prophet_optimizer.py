import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import optuna
from src.core.models.model_validation import calculate_metrics
from src.utils.dataframe_utils import train_test_split
from prophet import Prophet


def stop_when_no_improvement(study, trial):

    window = 20

    if len(study.trials) < window:
        return

    best_recent = min(t.value for t in study.trials[-window:])
    best_global = study.best_value

    if best_recent >= best_global:
        study.stop()


def prophet_optimizer(y, 
                      seasonal_periods,
                      freq,
                      n_trials=200):
    
    """Временная фейк версия"""
    
    threshold = seasonal_periods * 3
    if len(y) >= threshold:
        y_train, y_val = train_test_split(y, 0.1)
    else:
        y_train = y
        y_val = None

    def objective(trial):

        try:
            model = Prophet().fit(y_train)

            if y_val is not None:
                return calculate_metrics(y_val['y'].values, model.predict(y_val[['ds']])['yhat'].values)['WAPE']
            else:
                return model.aic
        except Exception as e:
            print(e)
            return 1e10

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_params = study.best_params

    if y_val is not None:
        ansamble_weight = study.best_value
    else:
        ansamble_weight = 1

    # for i in best_params:
    #     all_params[i] = best_params[i]
    # order = (all_params['p'], all_params['d'], all_params['q'])
    # seasonal_order = (all_params['P'], all_params['D'], all_params['Q'], m)
    # best_trend = all_params['trend']

    return (ansamble_weight)


