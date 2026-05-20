from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd
import math
from statsmodels.stats.diagnostic import acorr_ljungbox

def calculate_metrics(actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = mean_absolute_percentage_error(actual, pred)*100
        # Избегаем деления на ноль в MAPE
        wape = np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100
        return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape}


def penalized_wape(wape, model, min_train_size, use_params_penalty=True):
    """
    Применяет мультипликативный штраф к WAPE за автокорреляцию остатков,
    незначимые коэффициенты и избыточную сложность модели.
    """

    try:
        n_obs = len(model.resid)
        max_lag = min_train_size
        
        lb_df = acorr_ljungbox(model.resid[2:], lags=max_lag, return_df=True)
        min_lb_pvalue = lb_df['lb_pvalue'].min()
    except Exception as e:
        print(f"Error calculating Ljung-Box test: {e}")
        min_lb_pvalue = None

    k = len(model.params)
    
    if use_params_penalty == 'hard':
        wape *= 1 + (k * math.log(len(model.resid))) / len(model.resid)

    if use_params_penalty == 'soft':
        wape *= 1 + (2 * k) / len(model.resid)

    if use_params_penalty == 'mean':
        wape *= 1 + (((k * math.log(len(model.resid))) / len(model.resid) + (2 * k) / len(model.resid)) / 2)

    if min_lb_pvalue is not None and min_lb_pvalue < 0.15:
        print(f"Applying Ljung-Box penalty: min p-value = {min_lb_pvalue:.4f}")
        lb_penalty = 1 + (1.0 - (min_lb_pvalue / 0.15))
        wape *= lb_penalty

    print(f"Final penalized WAPE: {wape:.4f}, Params penalty: {'Yes' if use_params_penalty else 'No'}, Ljung-Box penalty applied: {'Yes' if min_lb_pvalue is not None and min_lb_pvalue < 0.15 else 'No'}, model params: {model.params}, min Ljung-Box p-value: {min_lb_pvalue:.4f}" if min_lb_pvalue is not None else f"Final penalized WAPE: {wape:.4f}, Params penalty: {'Yes' if use_params_penalty else 'No'}, Ljung-Box penalty applied: No, model params: {len(model.params)}, min Ljung-Box p-value: N/A")
    return wape


def multi_model_valid(pred_stat, y_test):
    actual_test = y_test
    metrics_list = []
    for model_name, data in pred_stat.items():

        # Берем первые len(actual_test) элементов, безопасно по позиции
        test_pred_series = data['test_pred']
        predicted_test = test_pred_series.iloc[:len(actual_test)].values if len(test_pred_series) >= len(actual_test) else test_pred_series.values
        
        # Убедимся что длины совпадают
        if len(predicted_test) != len(actual_test):
            # Если не совпадают - берем минимум
            min_len = min(len(predicted_test), len(actual_test))
            predicted_test = predicted_test[:min_len]
            actual_values = actual_test[:min_len]
        else:
            actual_values = actual_test
        
        test_m = calculate_metrics(actual_values, predicted_test)
        m = {}
        for k, v in test_m.items():
            m[f'Test_{k}'] = v
        
        metrics_list.append({'Model': model_name, **m})

    test_stat_df = pd.DataFrame(metrics_list).set_index('Model')

    column_order = sorted(test_stat_df.columns)
    test_stat_df = test_stat_df[column_order]
    
    return test_stat_df

def check_residuals_autocorrelation(fitted_model, max_lag, d_order=0):
    """
    Универсальная проверка остатков обученной модели на автокорреляцию.
    """
    try:
        residuals = fitted_model.resid[d_order:]
        
        # Определяем максимальный лаг для проверки

        if max_lag < 1:
            max_lag = 1
            
        # Тест Льюнга-Бокса
        lb_test = acorr_ljungbox(residuals, lags=[max_lag], return_df=True)
        p_value = lb_test['lb_pvalue'].values[0]
        
        # p-value < 0.05 означает, что есть автокорреляция (модель плохая)
        return p_value < 0.05
    except Exception as e:
        print(f"Error in autocorrelation check: {e}")
        return True

    except Exception as e:
        # Если тест по какой-то причине упал (например, слишком мало данных), 
        # лучше перестраховаться и вернуть True (отклонить модель)
        print(f"Ошибка при тесте Льюнга-Бокса: {e}")
        return True