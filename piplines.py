import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# арима 
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.seasonal import STL


#профет
from prophet import Prophet


#Эксп сглаживание
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#метрики качества
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, TransformerMixin


#акф и чакф 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#тесты на стационарность 
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron


#оптуна для подбора гиперпараметров эксп.сглаживания
import optuna


# всякие вспомогательные функции
# функция для построения графика динамики
def plot_dynamic(x, y, title = None):
    sns.lineplot(x = x, y = y)
    plt.xticks(rotation=90)
    if title:
        plt.title(label = title)

# функция для отрисовки акф и чакф
def acf_chacf(sr):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # ACF
    plot_acf(sr, lags=20, ax=axes[0], zero=False)
    # PACF
    plot_pacf(sr, lags=20, ax=axes[1], method='ywm', zero=False)
    
    plt.show()
# acf_chacf(df['Средняя цена'].dropna())

def lag_prepare(df, dict):
    df_lagged = df.copy()
    for i in dict:
        if type(dict[i]) is list:
            for t in dict[i]:
                if t == 0:
                    continue
                else:
                    df_lagged[i+str(t)] = df_lagged[i].shift(t)
            df_lagged = df_lagged.rename(columns = {i: i+'0'})
        else:
            df_lagged[i] = df_lagged[i].shift(dict[i])
            df_lagged = df_lagged.rename(columns = {i: i+str(dict[i])})
    return df_lagged

# функция для построения предсказаний с и без данных по верным предсказаниям
def pred(X, reg, y_test = None):
    X = sm.add_constant(X)
    y_pred = reg.predict(X)
    if y_test is not None:
        print(f"MAE (ср. абс. ошибка): {metrics.mean_absolute_error(y_test, y_pred):.3f}")
        print(f"MSE (ср. квадр. ошибка): {metrics.mean_squared_error(y_test, y_pred):.3f}")
        print(f"RMSE (корень из MSE): {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.3f}")
        print(f"R^2 (коэф. детерминации): {metrics.r2_score(y_test, y_pred):.3f}")
        
        fig = sns.lineplot(y_pred, label = 'Предсказание')
        fig = sns.lineplot(y_test, label = 'Факт')
        plt.legend(loc='upper right')
        plt.show()
    
        print("Остатки")
        res = sns.lineplot(y_test-y_pred)
        plt.show()

        
    return y_pred

# pred(X_test, reg, y_test)

# функция для построения регрессии с графиками факт/предсказание и остатками + VIF по факторам
def reg_with_graph(X, y, log = False, timeline = None):

    if timeline is None:
        timeline = X.index
        
    reg = LinearRegression()
    
    X = X
    X = sm.add_constant(X)


    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                              for i in range(len(X.columns))]
    
    print(vif_data)
    
    result = sm.OLS(y, X).fit()
    
    print(result.summary())
    if log == True:
        y_pred = np.exp(y - result.resid)
        y_fact = np.exp(y)
    else:
        y_pred = y - result.resid
        y_fact = y
        
    fig = sns.lineplot(x = timeline, y = y_fact, label = 'Факт')
    fig = sns.lineplot(x = timeline, y = y_pred, label = 'Предсказание')
    plt.show()

    print("Остатки")
    res = sns.lineplot(x = timeline, y = y_fact-y_pred)
    plt.show()

    print('АКФ и ЧАКФ остатков')
    acf_chacf(y_fact-y_pred)
    return result

def stationarity_check(series, ADF = True, KPSS = False, PP = False, pct_change = None):

    if pct_change:
        series_pct = (series.pct_change(pct_change)*100).dropna()
        
    result = pd.DataFrame()
    
    if ADF:
        result.loc['Исходный ряд', 'ADF'] = adfuller(series)[1]
        if pct_change:
            result.loc['Ряд по темпам прироста', 'ADF'] = adfuller(series_pct)[1]
        
    if KPSS:
        result.loc['Исходный ряд', 'KPSS'] = kpss(series)[1]
        if pct_change:
            result.loc['Ряд по темпам прироста', 'KPSS'] = kpss(series_pct)[1]
        
    if PP:
        result.loc['Исходный ряд', 'PP'] = PhillipsPerron(series, trend='c', lags=12).pvalue
        if pct_change:
            result.loc['Ряд по темпам прироста', 'PP'] = PhillipsPerron(series_pct, trend='c', lags=12).pvalue

    return result

def calculate_metrics(actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = mean_absolute_percentage_error(actual, pred)*100
        # Избегаем деления на ноль в MAPE
        wape = np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100
        return {"MAE": mae, "RMSE": rmse, "WAPE": wape, "MAPE": mape}


class InteractionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, lag_cols, cross_vars):
        self.lag_cols = lag_cols
        self.cross_vars = cross_vars
        self.means_ = {}

    def fit(self, X, y=None):
        # Обучаемся: запоминаем среднее для каждой лаговой колонки
        for col in self.lag_cols:
            self.means_[col] = X[col].mean()
        return self

    def transform(self, X):
        X_res = X.copy()
        
        for lag_col in self.lag_cols:
            # 1. Центрируем исходный лаг (чтобы снизить корреляцию с произведением)
            centered_lag = X_res[lag_col] - self.means_[lag_col]
            centered_name = f"{lag_col}_centered"
            X_res[centered_name] = centered_lag # Обновляем оригинал на центрированный
            
            for cross_var in self.cross_vars:
                # 2. Создаем пересечение: (центрированный лаг) * дамми
                new_col_name = f"{lag_col}_X_{cross_var}"
                X_res[new_col_name] = centered_lag * X_res[cross_var]

            X_res = X_res.drop(columns=[centered_name])
                
        return X_res

    def get_feature_names_out(self, input_features=None):
        # input_features — это список колонок, пришедший с предыдущего шага
        feature_names = list(input_features)
        for lag_col in self.lag_cols:
            for cross_var in self.cross_vars:
                feature_names.append(f"{lag_col}_X_{cross_var}")
        return np.array(feature_names)

def auto_correlation_test(res, seasonal_preiod, alpha = 0.05):
    max_lag = len(res)

    for i in range(1, 2*seasonal_preiod+1, 5):
        if sm.stats.acorr_ljungbox(res.resid, lags=[5], return_df=True)[1]>alpha:
            return True
    return False
        

def auto_arima_fit(y, seasonal_periods, n_trials=30):

    res = STL(y, period=seasonal_periods, robust=True).fit()
    
    y_deseason = y - res.seasonal

    def objective(trial):
        p = trial.suggest_int('p', 0, 6)
        q = trial.suggest_int('q', 0, 6)
        trend = trial.suggest_categorical('trend', ['n', 'c', 't', 'ct'])
        
        try:
            model = ARIMA(y_deseason, order=(p, 0, q), trend=trend).fit()

            res = y - model.fittedvalues
            if auto_correlation_test(res, seasonal_period):
                return 1e10
            return model.aic
        except:
            return 1e10

    study = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_order = (best_params['p'], 0, best_params['q'])
    best_trend = best_params['trend']

    print("Best params:", best_order, "trend:", best_trend)

    return best_order, best_trend

    
def fit_exp_sm(y, seasonal_periods):
    def objective(trial):
        trend = trial.suggest_categorical('trend', ['add', 'mul', 'additive', 'multiplicative', None])
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul', 'additive', 'multiplicative', None])
    
        if trend is not None:
            damped = trial.suggest_categorical('damped', [True, False])
        else:
            damped = False
        
        model = ExponentialSmoothing(endog = y,
                                     trend = trend,
                                     damped_trend = damped,
                                     seasonal = seasonal, 
                                     seasonal_periods = seasonal_periods).fit()
    
        # рассчитываем метрики оценки 
        res_exp = model.fittedvalues
    
        mape = calculate_metrics(y,res_exp)['MAPE']
        return mape
    return objective

def plot_forecast_results(pred_stat, y_train, y_test, value_name = None):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 8))
    
    # 1. Отрисовка исторических данных
    plt.plot(y_train.ds, y_train['y'], label='Train (История)', color='black', linewidth=2, alpha=0.3)
    
    # 2. Отрисовка реального теста (если есть)
    if y_test is not None:
        plt.plot(y_test.ds, y_test['y'], label='Test (Реальность)', color='black', linewidth=3, marker='o', markersize=4)

    # 3. Отрисовка прогнозов каждой модели из словаря
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Красивая палитра
    
    for (model_name, data), color in zip(pred_stat.items(), colors):
        test_pred = data['test_pred']
        
        # Если это ансамбль (массив), создаем индекс для него
        if isinstance(test_pred, np.ndarray):
            test_index = pd.date_range(start=y_train.index[-1], periods=len(test_pred)+1, freq=y_train.index.freq)[1:]
        else:
            test_index = test_pred.index
            
        plt.plot(test_index, test_pred, label=f'Прогноз: {model_name}', linestyle='--', color=color, linewidth=2)

    plt.title(f'Сравнение моделей прогнозирования {value_name}', fontsize=16)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Объем (ед./кв.м.)', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()


def multi_model_forecast(y, forecast_len, test_size = None, seasonal_preiod = None, freq = 'MS', graph  = True, value_name = None):

    """
    Функция для прогнозирования на основе нескольких "простых" моделей
    """
    pred_stat = {}
                       
    y.columns = ['ds', 'y']
    y.index.freq = freq

    split_index = int(len(y)*test_size)
    
    if test_size:
        y_train, y_test = y.iloc[:-split_index, :], y.iloc[-split_index:, :]
    else:
        y_train = y
        y_test = None


    #предварительные тесты
    print(stationarity_check(y_train['y'], ADF = True, KPSS = True, PP = True)) #тест на стационарность
    

    # визаульное представление ряда динамики
    plot_dynamic(x = y['ds'], y = y['y'])
    acf_chacf(y['y'])
    #обучение
    
    #фит пророка 
    m = Prophet()
    m.fit(y_train)
    
    future = m.make_future_dataframe(periods=forecast_len, freq = freq)
    real_future = future[-forecast_len:]
    forecast_p = m.predict(future)

    p_train = forecast_p.iloc[:len(y_train)]['yhat'].values
    p_test = forecast_p.iloc[len(y_train):len(y_train)+forecast_len]['yhat'].values

    pred_stat['prop'] = { 'model': m
                         ,'train_pred': pd.Series(p_train, index=y_train.ds)
                         ,'test_pred': pd.Series(p_test, index = real_future.ds)
                         ,'metrics': calculate_metrics(y_train['y'], p_train)}

    #фит аримы
    best_order, best_trend = auto_arima_fit(y_train['y'], seasonal_preiod)
    stlf = STLForecast(y_train['y'], ARIMA, model_kwargs=dict(order=best_order, trend=best_trend), period=seasonal_preiod, robust=True)
    stlf_res = stlf.fit()
    
    train_preds = stlf_res.get_prediction(start=y_train.index[0], end=y_train.index[-1]).predicted_mean
    
    pred_stat['ar'] = { 'model': stlf_res
                       ,'train_pred': pd.Series(train_preds.values, index=y_train.ds)
                       ,'test_pred':pd.Series(stlf_res.forecast(forecast_len).values, index = real_future.ds)
                       ,'metrics': calculate_metrics(y_train['y'], train_preds)}
    

    #фит экспоненты

    study = optuna.create_study()  
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(fit_exp_sm(y_train['y'], seasonal_preiod), n_trials=10) 

    exp_model = ExponentialSmoothing(endog = y_train['y'], seasonal_periods = seasonal_preiod, **study.best_params).fit()

    exp_pred = exp_model.forecast(forecast_len)

    pred_stat['exp'] = { 'model': exp_model
                        ,'train_pred': pd.Series(exp_model.fittedvalues.values, index=y_train.ds)
                        ,'test_pred': pd.Series(exp_pred.values, index = real_future.ds)
                        ,'metrics': calculate_metrics(y_train['y'], exp_model.fittedvalues)}


    maes = {name: info['metrics']['MAE'] for name, info in pred_stat.items()}
    inv_sum = sum(1.0 / v for v in maes.values())
    weights = {k: (1.0 / v) / inv_sum for k, v in maes.items()}
    
    ensemble_train = sum(np.array(pred_stat[name]['train_pred']) * weights[name] for name in pred_stat)
    ensemble_test = sum(np.array(pred_stat[name]['test_pred']) * weights[name] for name in pred_stat)

    
    pred_stat['Ensemble'] = {
        'train_pred': pd.Series(ensemble_train, index=y_train.ds),
        'test_pred': pd.Series(ensemble_test, index = real_future.ds),
        'metrics': calculate_metrics(y_train['y'], ensemble_train),
        'weights': weights
    }

    metrics_list = []
    for model_name, data in pred_stat.items():
        m = data['metrics'].copy()
        m['Model'] = model_name
        metrics_list.append(m)
    
    stat_df = pd.DataFrame(metrics_list).set_index('Model')

    if test_size:
        metrics_list = []
        for model_name, data in pred_stat.items():
            m = data['metrics'].copy()
            m = {f'Train_{k}': v for k, v in m.items()} 
            m['Model'] = model_name
            
            actual_test = y_test['y'].values
            predicted_test = data['test_pred'][:len(actual_test)]
            test_m = calculate_metrics(actual_test, predicted_test)
            for k, v in test_m.items():
                m[f'Test_{k}'] = v
            
            metrics_list.append(m)
        
        stat_df = pd.DataFrame(metrics_list).set_index('Model')
        
        column_order = sorted(stat_df.columns)
        stat_df = stat_df[column_order]
        
        display(stat_df)

    if graph:
        plot_forecast_results(pred_stat, y_train, y_test, value_name)
    
    return pred_stat, stat_dfel.order 

    
def fit_exp_sm(y, seasonal_periods):
    def objective(trial):
        trend = trial.suggest_categorical('trend', ['add', 'mul', 'additive', 'multiplicative', None])
        seasonal = trial.suggest_categorical('seasonal', ['add', 'mul', 'additive', 'multiplicative', None])
    
        if trend is not None:
            damped = trial.suggest_categorical('damped', [True, False])
        else:
            damped = False
        
        model = ExponentialSmoothing(endog = y,
                                     trend = trend,
                                     damped_trend = damped,
                                     seasonal = seasonal, 
                                     seasonal_periods = seasonal_periods).fit()
    
        # рассчитываем метрики оценки 
        res_exp = model.fittedvalues
    
        mape = calculate_metrics(y,res_exp)['MAPE']
        return mape
    return objective

def plot_forecast_results(pred_stat, y_train, y_test, value_name = None):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 8))
    
    # 1. Отрисовка исторических данных
    plt.plot(y_train.ds, y_train['y'], label='Train (История)', color='black', linewidth=2, alpha=0.3)
    
    # 2. Отрисовка реального теста (если есть)
    if y_test is not None:
        plt.plot(y_test.ds, y_test['y'], label='Test (Реальность)', color='black', linewidth=3, marker='o', markersize=4)

    # 3. Отрисовка прогнозов каждой модели из словаря
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Красивая палитра
    
    for (model_name, data), color in zip(pred_stat.items(), colors):
        test_pred = data['test_pred']
        
        # Если это ансамбль (массив), создаем индекс для него
        if isinstance(test_pred, np.ndarray):
            test_index = pd.date_range(start=y_train.index[-1], periods=len(test_pred)+1, freq=y_train.index.freq)[1:]
        else:
            test_index = test_pred.index
            
        plt.plot(test_index, test_pred, label=f'Прогноз: {model_name}', linestyle='--', color=color, linewidth=2)

    plt.title(f'Сравнение моделей прогнозирования {value_name}', fontsize=16)
    plt.xlabel('Дата', fontsize=12)
    plt.ylabel('Объем (ед./кв.м.)', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()


def multi_model_forecast(y, forecast_len, test_size = None, seasonal_preiod = None, freq = 'MS', graph  = True, value_name = None):

    """
    Функция для прогнозирования на основе нескольких "простых" моделей
    """
    pred_stat = {}
                       
    y.columns = ['ds', 'y']
    y.index.freq = freq

    split_index = int(len(y)*test_size)
    
    if test_size:
        y_train, y_test = y.iloc[:-split_index, :], y.iloc[-split_index:, :]
    else:
        y_train = y
        y_test = None


    #предварительные тесты
    print(stationarity_check(y_train['y'], ADF = True, KPSS = True, PP = True)) #тест на стационарность
    

    # визаульное представление ряда динамики
    plot_dynamic(x = y['ds'], y = y['y'])
    acf_chacf(y['y'])
    #обучение
    
    #фит пророка 
    m = Prophet()
    m.fit(y_train)
    
    future = m.make_future_dataframe(periods=forecast_len, freq = freq)
    real_future = future[-forecast_len:]
    forecast_p = m.predict(future)

    p_train = forecast_p.iloc[:len(y_train)]['yhat'].values
    p_test = forecast_p.iloc[len(y_train):len(y_train)+forecast_len]['yhat'].values

    pred_stat['prop'] = { 'model': m
                         ,'train_pred': pd.Series(p_train, index=y_train.ds)
                         ,'test_pred': pd.Series(p_test, index = real_future.ds)
                         ,'metrics': calculate_metrics(y_train['y'], p_train)}

    #фит аримы
    arima_params = auto_arima_fit(y_train['y'], seasonal_preiod)
    stlf = STLForecast(y_train['y'], ARIMA, model_kwargs=dict(order=arima_params), period=seasonal_preiod)
    stlf_res = stlf.fit()

    train_preds = stlf_res.get_prediction(start=y_train.index[0], end=y_train.index[-1]).predicted_mean
    
    pred_stat['ar'] = { 'model': stlf_res
                       ,'train_pred': pd.Series(train_preds.values, index=y_train.ds)
                       ,'test_pred':pd.Series(stlf_res.forecast(forecast_len).values, index = real_future.ds)
                       ,'metrics': calculate_metrics(y_train['y'], train_preds)}
    

    #фит экспоненты

    study = optuna.create_study()  
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(fit_exp_sm(y_train['y'], seasonal_preiod), n_trials=10) 

    exp_model = ExponentialSmoothing(endog = y_train['y'], seasonal_periods = seasonal_preiod, **study.best_params).fit()

    exp_pred = exp_model.forecast(forecast_len)

    pred_stat['exp'] = { 'model': exp_model
                        ,'train_pred': pd.Series(exp_model.fittedvalues.values, index=y_train.ds)
                        ,'test_pred': pd.Series(exp_pred.values, index = real_future.ds)
                        ,'metrics': calculate_metrics(y_train['y'], exp_model.fittedvalues)}


    maes = {name: info['metrics']['MAE'] for name, info in pred_stat.items()}
    inv_sum = sum(1.0 / v for v in maes.values())
    weights = {k: (1.0 / v) / inv_sum for k, v in maes.items()}
    
    ensemble_train = sum(np.array(pred_stat[name]['train_pred']) * weights[name] for name in pred_stat)
    ensemble_test = sum(np.array(pred_stat[name]['test_pred']) * weights[name] for name in pred_stat)

    
    pred_stat['Ensemble'] = {
        'train_pred': pd.Series(ensemble_train, index=y_train.ds),
        'test_pred': pd.Series(ensemble_test, index = real_future.ds),
        'metrics': calculate_metrics(y_train['y'], ensemble_train),
        'weights': weights
    }

    metrics_list = []
    for model_name, data in pred_stat.items():
        m = data['metrics'].copy()
        m['Model'] = model_name
        metrics_list.append(m)
    
    stat_df = pd.DataFrame(metrics_list).set_index('Model')

    if test_size:
        metrics_list = []
        for model_name, data in pred_stat.items():
            m = data['metrics'].copy()
            m = {f'Train_{k}': v for k, v in m.items()} 
            m['Model'] = model_name
            
            actual_test = y_test['y'].values
            predicted_test = data['test_pred'][:len(actual_test)]
            test_m = calculate_metrics(actual_test, predicted_test)
            for k, v in test_m.items():
                m[f'Test_{k}'] = v
            
            metrics_list.append(m)
        
        stat_df = pd.DataFrame(metrics_list).set_index('Model')
        
        column_order = sorted(stat_df.columns)
        stat_df = stat_df[column_order]
        
        display(stat_df)

    if graph:
        plot_forecast_results(pred_stat, y_train, y_test, value_name)
    
    return pred_stat, stat_df