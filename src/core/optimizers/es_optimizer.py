from statsmodels.tsa.holtwinters import ExponentialSmoothing
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.core.models.model_validation import calculate_metrics, check_residuals_autocorrelation, penalized_wape
from src.utils.dataframe_utils import train_test_split
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm

def stop_when_no_improvement(study, trial):

    window = 50
    
    if len(study.trials) < window:
        return
    
    if trial.number - study.best_trial.number >= window:
        print(f"Early stopping: no improvement in the last {window} trials.")
        study.stop()

def es_optimizer(y, trend= None, seasonal = None, damped = None, seasonal_periods = None, alpha = None, beta = None, gamma = None, n_trials=1000):
    all_params = {'trend': trend, 
                  'seasonal': seasonal, 
                  'damped': damped, 
                  'seasonal_periods': seasonal_periods,
                  'alpha': alpha,
                  'beta': beta,
                  'gamma': gamma}
    
    # TimeSeriesSplit параметры
    min_train_size = seasonal_periods * 2
    min_test_size = int(seasonal_periods)
    
    # Проверка минимального размера для CV
    if len(y) < min_train_size + min_test_size:
        threshold = seasonal_periods * 3
        if len(y) >= threshold:
            y_train, y_val = train_test_split(y, 0.1)
        else:
            y_train = y
            y_val = None
        use_cv = False
    else:
        use_cv = True
        n_folds = int((len(y) - min_train_size) // min_test_size)
        
        if n_folds >= 2:
            n_folds = min(n_folds, 5)
            y_val = 'cv'
            y_train = y.iloc[:-int(min_test_size)]
            y_val = y.iloc[-int(min_test_size):]
        else:
            use_cv = False
            y_train = y.iloc[:-int(min_test_size)]
            y_val = y.iloc[-int(min_test_size):]

    def objective(trial):
        trend = all_params['trend'] if all_params['trend'] != None else trial.suggest_categorical('trend', ['add', 'mul', None])
        seasonal = all_params['seasonal'] if all_params['seasonal'] != None else trial.suggest_categorical('seasonal', ['add', 'mul',  None])
        damped = all_params['damped'] if all_params['damped'] != None else trial.suggest_categorical('damped', [True, False, None])

        if damped and trend is None:
            return 1e10

        if trend == 'mul' and seasonal == 'mul':
            return 1e10
        
        alpha = all_params['alpha'] if all_params['alpha'] != None else trial.suggest_float('alpha', 0, 1)
        beta = all_params['beta'] if all_params['beta'] != None else trial.suggest_float('beta', 0, 1) if trend is not None else None
        gamma = all_params['gamma'] if all_params['gamma'] != None else trial.suggest_float('gamma', 0, 1) if seasonal is not None else None
        
        try:
            if use_cv:
                # TimeSeriesSplit для кросс-валидации
                tscv = TimeSeriesSplit(n_splits=n_folds, max_train_size=None, test_size=int(min_test_size), gap=0)
                cv_scores = []
                cv_raw_scores = []

                fit_kwargs = {
                    'smoothing_level': alpha,
                    'optimized': False
                }

                if trend is not None:
                    fit_kwargs['smoothing_trend'] = beta

                if seasonal is not None:
                    fit_kwargs['smoothing_seasonal'] = gamma
                
                for train_idx, test_idx in tscv.split(y):
                    if len(train_idx) < min_train_size:
                        continue
                    
                    y_cv_train = y.iloc[train_idx]
                    y_cv_test = y.iloc[test_idx]
                    try:
                        model = ExponentialSmoothing(endog=y_cv_train,
                                                 trend=trend,
                                                 damped_trend=damped,
                                                 seasonal=seasonal, 
                                                 seasonal_periods=seasonal_periods).fit(**fit_kwargs)
                        wape_cv = calculate_metrics(y_cv_test['y'].values, model.forecast(len(y_cv_test)))['WAPE']
                        cv_raw_scores.append(wape_cv)
                        # wape_cv = penalized_wape(wape_cv, model, min_train_size, use_params_penalty=False)
                        # cv_scores.append(wape_cv)
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        return 1e10
                    
                # if not cv_scores:
                #     return 1e10
                
                
                # mean_wape = np.mean(cv_scores)
                mean_raw_wape = np.mean(cv_raw_scores)
                trial.set_user_attr('raw_wape', mean_raw_wape)

                try:
                    full_fitted_model = ExponentialSmoothing(
                        endog=y,
                        trend=trend,
                        damped_trend=damped,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods
                    ).fit(**fit_kwargs)
                    

                    final_wape = penalized_wape(mean_raw_wape, full_fitted_model, min_train_size, use_params_penalty='soft')
                        
                except Exception as e:
                    print(f"Error in full model fit or penalty check for ETS: {e}")
                    return 1e10
                
                return final_wape
            else:
                fit_kwargs = {
                    'smoothing_level': alpha,
                    'optimized': False
                }

                if trend is not None:
                    fit_kwargs['smoothing_trend'] = beta

                if seasonal is not None:
                    fit_kwargs['smoothing_seasonal'] = gamma

                model = ExponentialSmoothing(endog=y_train,
                                         trend=trend,
                                         damped_trend=damped,
                                         seasonal=seasonal, 
                                         seasonal_periods=seasonal_periods).fit(**fit_kwargs)
                
                if y_val is not None:
                    wape = calculate_metrics(y_val['y'].values, model.forecast(len(y_val)))['WAPE']
                    trial.set_user_attr('raw_wape', wape)
                    
                    try:
                        full_fitted_model = ExponentialSmoothing(
                            endog=y,
                            trend=trend,
                            damped_trend=damped,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        ).fit(**fit_kwargs)
                        
                        final_wape = penalized_wape(wape, full_fitted_model, min_train_size, use_params_penalty='soft')
                        
                    except Exception as e:
                        print(f"Error in full model fit or penalty check for ETS (no CV): {e}")
                        return 1e10
                        
                    return final_wape
                else:
                    return model.aic
                    
        except Exception as e:
            print(f"Error occurred: {e}")
            return 1e10
    
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, callbacks=[stop_when_no_improvement])

    best_params = study.best_params


    for i in best_params:
        all_params[i] = best_params[i]
    
    trend = all_params['trend']
    seasonal = all_params['seasonal']
    damped_trend = all_params['damped']
    seasonal_periods = all_params['seasonal_periods']
    alpha = all_params['alpha']
    beta = all_params['beta']
    gamma = all_params['gamma']

    print(best_params)
    try:
        print(trend, seasonal, damped_trend, seasonal_periods, study.best_trial.user_attrs['raw_wape'], alpha, beta, gamma)
        return (trend, seasonal, damped_trend, seasonal_periods, study.best_trial.user_attrs['raw_wape'], alpha, beta, gamma)
    except:
        print(trend, seasonal, damped_trend, seasonal_periods, study.best_value, alpha, beta, gamma)
        return (trend, seasonal, damped_trend, seasonal_periods, study.best_value, alpha, beta, gamma)
