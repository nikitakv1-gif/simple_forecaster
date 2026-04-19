import numpy as np
import pandas as pd
from ..models.model_validation import calculate_metrics

def ensemble_calculate(pred_stat, y_train, y_test):

    maes = {name: info['ensemble_weight'] for name, info in pred_stat.items()}
    inv_sum = sum(1.0 / v for v in maes.values())
    weights = {k: (1.0 / v) / inv_sum for k, v in maes.items()}
    
    ensemble_train = sum(np.array(pred_stat[name]['train_pred']) * weights[name] for name in pred_stat)
    ensemble_test = sum(np.array(pred_stat[name]['test_pred']) * weights[name] for name in pred_stat)

    
    pred_stat['Ensemble'] = {
        'train_pred': pd.Series(ensemble_train, index=y_train.ds),
        'test_pred': pd.Series(ensemble_test, index = y_test.ds),
        'weights': weights
    }
    return pred_stat