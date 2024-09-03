
import pandas as pd
import numpy as np
import sqlite3 as sql
import matplotlib.pyplot as plt
import plotly_express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns
import datetime
from lightgbm import LGBMRegressor
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error, r2_score
import funciones as func

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

def preprocesar_df_futuro(df_actual):
    fecha_maxima = df_actual.index.max()
    una_semana_mas = fecha_maxima + pd.DateOffset(days=5)
    future = pd.date_range(fecha_maxima+ pd.DateOffset(days=1), una_semana_mas, freq='D')
    future_exog = pd.DataFrame(index=future)
    future = pd.concat([df_actual, future_exog])
    future.drop(columns=['reward_3.125', 'reward_12.5', 'reward_25.0', 'reward_50.0', 'mes_1',
       'mes_2', 'mes_3', 'mes_4', 'mes_5', 'mes_6', 'mes_7', 'mes_8', 'mes_9',
       'mes_10', 'mes_11', 'mes_12'],inplace=True)
    future['mes'] = future.index.month
    future = func.calcular_recompensa_y_cuenta_regresiva_1d(df_1d=future)
    future = pd.get_dummies(future, columns=['reward', 'mes'], dtype=int)
    fecha = future['index'][-1:].values[0]
    ultima_fecha = pd.to_datetime(fecha)
    num_dias = len(future)
    fechas = pd.date_range(end=ultima_fecha, periods=num_dias)
    future['index'] = fechas
    future.index = fechas
    return future

# Function to load and prepare data
def load_and_prepare_data(db_path):
    conn = sql.connect(db_path)
    df_1d = pd.read_sql_query("SELECT * FROM btc_1d", conn)
    conn.close()
    df_1d.drop(columns=[
        'open', 'high', 'low', 'volume', 'var', 'return', 'diff', 'volatility', 
        'rsi_14', 'rsi_28', 'rsi_14_shifted', 'rsi_28_shifted', 'ma_5', 'ma_20', 
        'ma_100', 'MiddleBand', 'UpperBand', 'LowerBand', 'K', 'D', 'close_shifted', 
        'TR', 'ATR', 'TP', 'CCI', 'lag1_TR', 'lag2_TR', 'lag1_ATR', 'lag2_ATR'], inplace=True)
    fecha = df_1d['date'][-1:].values[0]
    ultima_fecha = pd.to_datetime(fecha)
    num_dias = len(df_1d)
    fechas = pd.date_range(end=ultima_fecha, periods=num_dias)
    df_1d['date'] = fechas
    df_1d.index = fechas
    df_1d['mes'] = df_1d.index.month
    data = pd.get_dummies(df_1d, columns=['reward', 'mes'], dtype=int)
    return data, df_1d

# Function to create and train the model
def create_and_train_model(data, lags=15, steps=5):
    exog = [column for column in data.columns if column.startswith(('reward', 'mes'))]
    exog.extend(['countdown_halving'])
    forecaster = ForecasterAutoreg(regressor=LGBMRegressor(random_state=42), lags=lags)
    forecaster.fit(y=data['close'], exog=data[exog])
    horizonte = steps
    df_and_future = preprocesar_df_futuro(df_actual=data)
    predicciones = forecaster.predict(steps=horizonte, exog=df_and_future[exog].iloc[-horizonte:])
    
    return predicciones

# Main script execution
if __name__ == "__main__":
    pass
    #data, df_1d = load_and_prepare_data('Data/db/btc.db')
    #predictions = create_and_train_model(data)
    #print(predictions)
