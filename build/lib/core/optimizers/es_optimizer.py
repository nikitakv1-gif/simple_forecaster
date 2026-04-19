from statsmodels.tsa.holtwinters import ExponentialSmoothing
import optuna
import pandas as pd
from ..models.model_validation import calculate_metrics
from ...utils.dataframe_utils import train_test_split

def fit_exp_sm(y, trend= None, seasonal = None, damped = None, seasonal_periods = None):
    all_params = {'trend': trend, 
                  'seasonal': seasonal, 
                  'damped': damped, 
                  'seasonal_periods': seasonal_period}
    
    threshold = seasonal_periods * 3
    if len(y) >= threshold:
        y_train, y_val = train_test_split(y, 0.2)
    else:
        y_train = y
        y_val = None

    def objective(trial):
        trend = trend if trend != None else trial.suggest_categorical('trend', ['add', 'mul', 'additive', 'multiplicative', None])
        seasonal = seasonal if seasonal != None else trial.suggest_categorical('seasonal', ['add', 'mul', 'additive', 'multiplicative', None])
    
        if trend is not None:
            damped = damped if damped != None else trial.suggest_categorical('damped_trend', [True, False, None])
        else:
            damped = damped if damped != None else trial.suggest_categorical('damped_trend', [False, None])
        
        try:
            model = ExponentialSmoothing(endog = y_train,
                                     trend = trend,
                                     damped_trend = damped,
                                     seasonal = seasonal, 
                                     seasonal_periods = seasonal_periods).fit()
            if y_val != None:
                return calculate_metrics(model.forecast(len(y_val)), y_val)['WAPE']
            else:
                return model.aic
        except:
            return 1e10
    
    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=30)

    best_params = study.best_params

    if y_val != None:
        ansamble_weight = study.best_value
    else:
        ansamble_weight = 1

    for i in best_params:
        all_params[i] = best_params[i]
    
    trend = all_params['trend']
    seasonal = all_params['seasonal']
    damped_trend = all_params['damped']
    seasonal_periods = all_params['seasonal_periods']

    return (trend, seasonal, damped_trend, seasonal_periods, ansamble_weight)
