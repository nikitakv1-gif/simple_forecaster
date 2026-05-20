import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL
import optuna
from sklearn.model_selection import TimeSeriesSplit
from src.core.models.model_validation import calculate_metrics, check_residuals_autocorrelation, penalized_wape
from src.utils.dataframe_utils import train_test_split
from statsmodels.stats.diagnostic import acorr_ljungbox


def stop_when_no_improvement(study, trial):

    window = 50
    
    if len(study.trials) < window:
        return
    
    if trial.number - study.best_trial.number >= window:
        print(f"Early stopping: no improvement in the last {window} trials.")
        study.stop()

def stl_arima_optimizer(y, 
                   seasonal_periods, 
                   arima_params = None,
                   trend = None,
                   n_trials=200):

    # TimeSeriesSplit параметры
    min_train_size = seasonal_periods * 2
    min_test_size = int(seasonal_periods)
    
    # Проверка минимального размера для CV
    if len(y) < min_train_size + min_test_size:
        threshold = seasonal_periods * 2
        if len(y) >= threshold:
            y_train, y_val = train_test_split(y, seasonal_periods)
        else:
            y_train = y
            y_val = None
        use_cv = False
    else:
        use_cv = True
        n_folds = (len(y) - min_train_size) // min_test_size
        
        if n_folds >= 2:
            n_folds = min(n_folds, 5) 
            y_train = y.iloc[:-int(min_test_size)]
            y_val = y.iloc[-int(min_test_size):]
        else:
            use_cv = False
            y_train = y.iloc[:-min_test_size]
            y_val = y.iloc[-min_test_size:]
        
    if arima_params == None:
        p, q, d = None, None, None 
    else:
        p, q, d = arima_params

    all_params = {'p' : p, 'd': d, 'q': q,
                  'm': seasonal_periods, 'trend': trend}
    
    def objective(trial):

        p = all_params['p'] if all_params['p'] != None else trial.suggest_int('p', 0, 3)
        d = all_params['d'] if all_params['d'] != None else trial.suggest_int('d', 0, 1)
        q = all_params['q'] if all_params['q'] != None else trial.suggest_int('q', 0, 3)
        trend_param = all_params['trend'] if all_params['trend'] != None else trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])

        if d >= 2 and trend_param != 'n':
            return 1e10
        elif d == 1 and trend_param not in ['n', 't']:
            return 1e10
        
        try:
            if use_cv:
                # TimeSeriesSplit для кросс-валидации
                tscv = TimeSeriesSplit(n_splits=n_folds, max_train_size=None, test_size=min_test_size, gap=0)
                cv_scores = []
                cv_raw_scores = []
                
                for train_idx, test_idx in tscv.split(y):
                    if len(train_idx) < min_train_size:
                        continue
                    
                    y_cv_train = y.iloc[train_idx]
                    y_cv_test = y.iloc[test_idx]
                    
                    try:
                        # Применяем STL на каждом фолде
                        res = STL(y_cv_train['y'], period=seasonal_periods, robust=True).fit()
                        y_deseason = y_cv_train['y'] - res.seasonal
                        
                        # Предсказываем сезонность на тестовом наборе
                        seasonal_pattern = res.seasonal[-seasonal_periods:]
                        future_season = np.tile(seasonal_pattern, int(np.ceil(len(y_cv_test) / seasonal_periods)))[:len(y_cv_test)]
                        
                        model = ARIMA(y_deseason, order=(p, d, q), trend=trend_param).fit()
                        pred_arima = model.forecast(len(y_cv_test))
                        pred = pred_arima.values + future_season
                        
                        wape = calculate_metrics(y_cv_test['y'].values, pred)['WAPE']
                        cv_raw_scores.append(wape)
                        # wape = penalized_wape(wape, model, min_train_size, use_params_penalty=False)
                        # cv_scores.append(wape)
                    except Exception as e:
                        print(e)
                        return 1e10
                
                # if not cv_scores:
                #     return 1e10

                # mean_wape = np.mean(cv_scores)
                mean_raw_wape = np.mean(cv_raw_scores)
                trial.set_user_attr('raw_wape', mean_raw_wape)
                
                try:
                    # Разлагаем полный ряд данных, которые пришли в оптимизатор
                    res_full = STL(y['y'], period=seasonal_periods, robust=True).fit()
                    y_deseason_full = y['y'] - res_full.seasonal
                    
                    # Строим готовую модель на всем трейне с текущими p, d, q
                    full_fitted_model = ARIMA(y_deseason_full, order=(p, d, q), trend=trend_param).fit()
                    
                    # # Передаем готовую модель в фильтр
                    # has_autocorrelation = check_residuals_autocorrelation(
                    #     fitted_model=full_fitted_model,
                    #     max_lag = min_train_size,
                    #     d_order=d
                    # )
                    
                    final_wape = penalized_wape(mean_raw_wape, full_fitted_model, min_train_size, use_params_penalty='mean')
                        
                except Exception as e:
                    print(f"Error in residuals check: {e}")
                    return 1e10

                # if has_autocorrelation:
                #     return 1e10
                return final_wape 
            
            else:
                # Обычный подход если данных мало
                res = STL(y_train['y'], period=seasonal_periods, robust=True).fit()
                y_deseason = y_train['y'] - res.seasonal
                
                seasonal_pattern = res.seasonal[-seasonal_periods:]
                future_season = np.tile(seasonal_pattern, int(np.ceil(len(y_val) / seasonal_periods)))[:len(y_val)] if y_val is not None else None
                
                model = ARIMA(y_deseason, order=(p, d, q), trend=trend_param).fit()
                
                if y_val is not None:
                    pred_arima = model.forecast(len(y_val))
                    pred = pred_arima.values + future_season

                    wape = calculate_metrics(y_val['y'].values, pred)['WAPE']
                    trial.set_user_attr('raw_wape', wape)
                    
                    try:
                        # Разлагаем полный ряд данных, которые пришли в оптимизатор
                        res_full = STL(y['y'], period=seasonal_periods, robust=True).fit()
                        y_deseason_full = y['y'] - res_full.seasonal
                        
                        # Строим готовую модель на всем трейне с текущими p, d, q
                        full_fitted_model = ARIMA(y_deseason_full, order=(p, d, q), trend=trend_param).fit()
                        
                        # # Передаем готовую модель в фильтр
                        # has_autocorrelation = check_residuals_autocorrelation(
                        #     fitted_model=full_fitted_model,
                        #     max_lag = min_train_size,
                        #     d_order=d
                        # )
                        
                        wape = penalized_wape(wape, full_fitted_model, min_train_size, use_params_penalty='mean')
                            
                    except Exception as e:
                        print(f"Error in residuals check: {e}")
                        return 1e10
                                
                    return wape
                else:
                    return model.aic
        except Exception as e:
            print(e)
            return 1e10
        

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_trial = study.best_trial
    
    p, d, q = best_trial.params['p'], best_trial.params['d'], best_trial.params['q']
    best_trend = best_trial.params['trend']
    
    try:
        return ((p, d, q), best_trend, study.best_trial.user_attrs['raw_wape'])
    except:
        return ((p, d, q), best_trend, study.best_value)