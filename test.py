from src.pipelines.simple_forecaster import multi_model_forecast
import pandas as pd

y = pd.read_excel('example.xlsx')
y['unsold/sold'] = y['unsold']/y['sold']
y = y[['Дата', 'sold']]
forecast_len = 24
test_size = 6
freq = 'MS'
model_list = ['stl_arima', 'es', 'ensemble']


forecast = multi_model_forecast(
                                y,
                                forecast_len, 
                                test_size = test_size, 
                                freq = 'MS',
                                model_list = model_list,
                                graph = True)

forecast['test_res']
forecast['plots'].show()