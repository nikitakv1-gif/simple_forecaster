from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import pandas as pd

def calculate_metrics(actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = mean_absolute_percentage_error(actual, pred)*100
        # Избегаем деления на ноль в MAPE
        wape = np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100
        return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape}

def multi_model_valid(pred_stat, y_test):
    actual_test = y_test
    metrics_list = []
    for model_name, data in pred_stat.items():

        predicted_test = data['test_pred'][:len(actual_test)].values
        test_m = calculate_metrics(actual_test, predicted_test)
        m = {}
        for k, v in test_m.items():
            m[f'Test_{k}'] = v
        
        metrics_list.append({'Model': model_name, **m})

    test_stat_df = pd.DataFrame(metrics_list).set_index('Model')

    column_order = sorted(test_stat_df.columns)
    test_stat_df = test_stat_df[column_order]
    
    return test_stat_df
