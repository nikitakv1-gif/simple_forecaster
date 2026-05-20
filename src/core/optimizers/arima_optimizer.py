import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import optuna
from sklearn.model_selection import TimeSeriesSplit
from src.core.models.model_validation import calculate_metrics, penalized_wape
from src.utils.dataframe_utils import train_test_split


def stop_when_no_improvement(study, trial):

    window = 20

    if len(study.trials) < window:
        return

    def trial_wape(t):
        if t.state != optuna.trial.TrialState.COMPLETE:
            return np.inf
        return t.values[1] if t.values is not None else np.inf

    recent = [trial_wape(t) for t in study.trials[-window:]]
    best_recent = min(recent) if recent else np.inf
    best_global = min((trial_wape(t) for t in study.trials), default=np.inf)

    if best_recent >= best_global:
        study.stop()


def arima_optimizer(y, 
                    seasonal_periods,
                    arima_params = None, 
                    seasonal_params = None, 
                    trend_params = None, 
                    n_trials=200):
    
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
    
    # TimeSeriesSplit параметры
    min_train_size = seasonal_periods * 2
    min_test_size = int(seasonal_periods * 0.5)
    
    # Проверка минимального размера для CV
    if len(y) < min_train_size + min_test_size:
        # Если данных недостаточно, используем простой split
        threshold = seasonal_periods * 3
        if len(y) >= threshold:
            y_train, y_val = train_test_split(y, m)
        else:
            y_train = y
            y_val = None
        use_cv = False
    else:
        use_cv = True
        n_folds = int((len(y) - min_train_size) // min_test_size)
        
        if n_folds >= 2:
            n_folds = min(n_folds, 5)
        else:
            use_cv = False
            y_train = y.iloc[:-int(min_test_size)]
            y_val = y.iloc[-int(min_test_size):]

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
            return 1e10, 1e10
        elif d + D == 1 and trend not in ['n', 't']:
            return 1e10, 1e10

        try:
            if use_cv:
                # TimeSeriesSplit для кросс-валидации
                tscv = TimeSeriesSplit(n_splits=n_folds, max_train_size=None, test_size=int(min_test_size), gap=0)
                cv_scores_aic = []
                cv_scores_raw_wape = []
                cv_scores_penalized = []
                
                for train_idx, test_idx in tscv.split(y):
                    if len(train_idx) < min_train_size:
                        continue
                    
                    y_cv_train = y.iloc[train_idx]
                    y_cv_test = y.iloc[test_idx]
                    
                    try:
                        model = ARIMA(y_cv_train, order=(p, d, q), seasonal_order=(P, D, Q, m), trend=trend).fit()
                        cv_scores_aic.append(model.aic)
                        raw_wape_cv = calculate_metrics(y_cv_test['y'].values, model.forecast(len(y_cv_test)))['WAPE']
                        cv_scores_raw_wape.append(raw_wape_cv)
                        cv_scores_penalized.append(penalized_wape(raw_wape_cv, model, min_train_size, use_significance=False))
                    except:
                        return 1e10, 1e10
                
                if not cv_scores_aic:
                    return 1e10, 1e10
                    
                aic_value = np.mean(cv_scores_aic)
                raw_wape_value = np.mean(cv_scores_raw_wape)
                penalized_wape_value = np.mean(cv_scores_penalized)
                trial.set_user_attr('raw_wape', raw_wape_value)
                wape_value = raw_wape_value  # Возвращаем честный WAPE, не пенализированный
            else:
                # Обычный подход если данных мало
                model = ARIMA(y_train, order=(p, d, q), seasonal_order=(P, D, Q, m), trend=trend).fit()
                aic_value = model.aic
                if y_val is not None:
                    raw_wape_value = calculate_metrics(y_val['y'].values, model.forecast(len(y_val)))['WAPE']
                    wape_value = raw_wape_value  # Возвращаем честный WAPE, не пенализированный
                    trial.set_user_attr('raw_wape', raw_wape_value)
                else:
                    wape_value = model.bic

            return aic_value, wape_value
        except Exception as e:
            print(e)
            return 1e10, 1e10

    study = optuna.create_study(directions=['minimize', 'minimize'], sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_trial = min(study.best_trials, key=lambda t: t.values[1]) # Сортируем по WAPE
    best_params = best_trial.params
    ansamble_weight = best_trial.user_attrs.get('raw_wape', best_trial.values[1])

    for i in best_params:
        all_params[i] = best_params[i]
    order = (all_params['p'], all_params['d'], all_params['q'])
    seasonal_order = (all_params['P'], all_params['D'], all_params['Q'], m)
    best_trend = all_params['trend']

    
    return (order, seasonal_order, best_trend, ansamble_weight)


