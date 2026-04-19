from src.pipelines.simple_forecaster import multi_model_forecast
import pandas as pd

y = pd.read_excel('example.xlsx')
y = y[['Дата', 'unsold']]
forecast_len = 24
test_size = 12
freq = 'MS'
model_list = ['prophet', 'stl_arima', 'es', 'ensemble']


forecast = multi_model_forecast(
                                y,
                                forecast_len, 
                                test_size = test_size, 
                                freq = 'MS',
                                model_list = model_list,
                                graph = True)

forecast['test_res']
forecast['plots'].show()