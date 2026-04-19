import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_forecast_results(pred_stat, y_train, y_test=None, value_name=None):

    fig = go.Figure()

    # 1. Train
    fig.add_trace(
        go.Scatter(
            x=y_train.ds,
            y=y_train['y'],
            mode='lines',
            name='Train (История)',
            line=dict(color='black', width=2),
            opacity=0.4
        )
    )

    # 2. Test
    if y_test is not None:
        fig.add_trace(
            go.Scatter(
                x=y_test.ds,
                y=y_test['y'],
                mode='lines+markers',
                name='Test (Реальность)',
                line=dict(color='black', width=2),
                marker=dict(size=5)
            )
        )

    # 3. Forecasts
    for model_name, data in pred_stat.items():
        test_pred = data['test_pred']

        # индекс прогнозов
        if isinstance(test_pred, np.ndarray):
            test_index = pd.date_range(
                start=y_train.index[-1],
                periods=len(test_pred) + 1,
                freq=y_train.index.freq
            )[1:]
        else:
            test_index = test_pred.index

        fig.add_trace(
            go.Scatter(
                x=test_index,
                y=test_pred,
                mode='lines',
                name=f'Прогноз: {model_name}',
                line=dict(dash='dash')
            )
        )

    fig.update_layout(
        title=f'Сравнение моделей прогнозирования {value_name}' if value_name else 'Forecast comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        template='plotly_white',
        legend=dict(orientation="h")
    )

    return fig