
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
    #print("future antes del future",future)
    future_exog = pd.DataFrame(index=future)
    #print("future_exog",future_exog)
    df_and_future = pd.concat([df_actual, future_exog])
    #print("future despues de concat:",future.head(1))
    
    df_and_future.drop(columns=['reward_3.125', 'reward_12.5', 'reward_25.0', 'reward_50.0', 'mes_1',
       'mes_2', 'mes_3', 'mes_4', 'mes_5', 'mes_6', 'mes_7', 'mes_8', 'mes_9',
       'mes_10', 'mes_11', 'mes_12'],inplace=True)
    df_and_future['mes'] = df_and_future.index.month
    
    df_and_future = func.calcular_recompensa_y_cuenta_regresiva_1d_future(df_1d=df_and_future)
    df_and_future = pd.get_dummies(df_and_future, columns=['reward', 'mes'], dtype=int)
    #print("index de df_and_future",df_and_future.index)
    #print("future",future.tail(3))
    
    #print("future:",df_and_future[['reward_3.125','mes_1']].tail(3))
    #print("future",future.tail(3))
    #df_and_future.drop(columns='index',inplace=True)
    df_and_future.reset_index(inplace=True)
    fecha = df_and_future['index'][-1:].values[0]
    #print(fecha)
    ultima_fecha = pd.to_datetime(fecha)
    num_dias = len(df_and_future)
    fechas = pd.date_range(end=ultima_fecha, periods=num_dias)
    df_and_future['index'] = fechas
    df_and_future.index = fechas
    return df_and_future


def preprocesar_df_futuro_sin_exog(df_actual):
    fecha_maxima = df_actual.index.max()
    una_semana_mas = fecha_maxima + pd.DateOffset(days=5)
    future = pd.date_range(fecha_maxima+ pd.DateOffset(days=1), una_semana_mas, freq='D')
    #print("future antes del future",future)
    future_exog = pd.DataFrame(index=future)
    #print("future_exog",future_exog)
    df_and_future = pd.concat([df_actual, future_exog])
    #print("future despues de concat:",future.head(1))
    
    df_and_future.drop(columns=['reward_3.125', 'reward_12.5', 'reward_25.0', 'reward_50.0', 'mes_1',
       'mes_2', 'mes_3', 'mes_4', 'mes_5', 'mes_6', 'mes_7', 'mes_8', 'mes_9',
       'mes_10', 'mes_11', 'mes_12'],inplace=True)
    df_and_future['mes'] = df_and_future.index.month
    
    df_and_future = func.calcular_recompensa_y_cuenta_regresiva_1d_future(df_1d=df_and_future)
    df_and_future = pd.get_dummies(df_and_future, columns=['reward', 'mes'], dtype=int)
    #print("index de df_and_future",df_and_future.index)
    #print("future",future.tail(3))
    
    #print("future:",df_and_future[['reward_3.125','mes_1']].tail(3))
    #print("future",future.tail(3))
    #df_and_future.drop(columns='index',inplace=True)
    df_and_future.reset_index(inplace=True)
    fecha = df_and_future['index'][-1:].values[0]
    #print(fecha)
    ultima_fecha = pd.to_datetime(fecha)
    num_dias = len(df_and_future)
    fechas = pd.date_range(end=ultima_fecha, periods=num_dias)
    df_and_future['index'] = fechas
    df_and_future.index = fechas
    return df_and_future



# Function to load and prepare data
def load_and_prepare_data(db_path):
    conn = sql.connect(db_path)
    df_1d = pd.read_sql_query("SELECT * FROM btc_1d", conn)
    conn.close()
    df_1d.drop(columns=[
        'open', 'high', 'low', 'volume', 'return', 'diff', 'volatility', 
        'rsi', 'ma_5', 'ma_20', 'ma_100', 'MiddleBand', 'UpperBand', 'LowerBand', 'K', 'D', 
         'TP', 'CCI'], inplace=True)
    fecha = df_1d['date'][-1:].values[0]
    ultima_fecha = pd.to_datetime(fecha)
    num_dias = len(df_1d)
    fechas = pd.date_range(end=ultima_fecha, periods=num_dias)
    df_1d['date'] = fechas
    df_1d.index = fechas
    df_1d['mes'] = df_1d.index.month
    data = pd.get_dummies(df_1d, columns=['reward', 'mes'], dtype=int)
    return data

def load_and_prepare_data_sin_exog(db_path,temporalidad):
    conn = sql.connect(db_path)
    data = pd.read_sql_query(f"""SELECT date, close, volume, volatility, rsi, ma_5, ma_20, ma_100, CCI, K, D,
               MiddleBand, UpperBand, LowerBand FROM btc_{temporalidad}""", conn)
    conn.close()
    print("data en model_script",data.tail(3))
    #data.drop(columns=[
    #    'open', 'high', 'low', 'volume', 'return', 'diff', 'volatility', 
    #    'rsi', 'ma_5', 'ma_20', 'ma_100', 'MiddleBand', 'UpperBand', 'LowerBand', 'K', 'D', 
    #     'TP', 'CCI'], inplace=True)
    return data

# Function to create and train the model
def create_and_train_model(data, lags=[1,30,90,180], steps=5):
    exog = [column for column in data.columns if column.startswith(('reward', 'mes'))]
    exog.extend(['countdown_halving'])
    forecaster = ForecasterAutoreg(regressor=LGBMRegressor(random_state=42), lags=lags)
    #print(data['close'].isna().sum())
    data.fillna(0,inplace=True)
    #print(data['close'].tail(4))
    #data.dropna(axis=1,inplace=True)
    #print(data.tail(2))
    forecaster.fit(y=data['close'], exog=data[exog])
    horizonte = steps
    df_and_future = preprocesar_df_futuro(df_actual=data)
    predicciones = forecaster.predict(steps=horizonte, exog=df_and_future[exog].iloc[-horizonte:])
    return predicciones

#def create_and_train_model(data, lags=[1,30,90,180], steps=5):
#    forecaster = ForecasterAutoreg(regressor=LGBMRegressor(random_state=42, verbose=-1), lags=lags)
#    data.fillna(0,inplace=True)
#    data = reordenar_fechas(data)
#    return None

def reordenar_fechas(data,temporalidad):
    fecha = data['date'][-1:].values[0]
    ultima_fecha = pd.to_datetime(fecha)
    num_instancias = len(data)
    if (temporalidad in ['4h','4H']):
        fechas = pd.date_range(end=ultima_fecha, periods=num_instancias,freq='4H')
    elif (temporalidad in ['1h','1H']):
        fechas = pd.date_range(end=ultima_fecha, periods=num_instancias,freq='1H')
    elif (temporalidad in ['5m','5M']):
        fechas = pd.date_range(end=ultima_fecha, periods=num_instancias,freq='5min')
    elif (temporalidad in ['1d','1D']):
        fechas = pd.date_range(end=ultima_fecha, periods=num_instancias,freq='1D')
    
    data['date'] = fechas
    data.index = fechas
    return data

def split_data(data,temporalidad):
    last_date = data['date'][-1:].values[0]
    if (temporalidad in ['4h','4H']):
        fin_train = last_date - pd.DateOffset(days=30)
    elif (temporalidad in ['1h','1H']):
        fin_train = last_date - pd.DateOffset(days=5)
    elif (temporalidad in ['5m','5M']):
        fin_train = last_date - pd.DateOffset(minutes=240)
    elif (temporalidad in ['1d','1D']):
        fin_train = last_date - pd.DateOffset(days=180)
    return fin_train

def create_train_model_sin_exog(data, lags=[1,30,90,180],steps=5,temporalidad=None):
    forecaster = ForecasterAutoreg(regressor=LGBMRegressor(random_state=42, verbose=-1), lags=lags)
    data.fillna(0,inplace=True)
    data = reordenar_fechas(data,temporalidad=temporalidad)
    data['date'] = pd.to_datetime(data['date'])
    last_date = data['date'][-1:].values[0]
    first_date = data['date'][:1].values[0]
    inicio_train = first_date
    #fin_train = last_date - pd.DateOffset(days=30)
    fin_train = split_data(data=data,temporalidad=temporalidad)
    inicio_train = pd.to_datetime(inicio_train)
    formatted_date_inicio = inicio_train.strftime('%Y-%m-%d %H:%M:%S')
    fin_train = pd.to_datetime(fin_train)
    metrica, predicciones = backtesting_forecaster(
        forecaster         = forecaster,
        y                  = data.loc[formatted_date_inicio:, 'close'],
        initial_train_size = len(data.loc[inicio_train:fin_train]),
        fixed_train_size   = True,
        steps              = 1,
        refit              = True,
        metric             = 'mean_absolute_percentage_error',
        verbose            = False,
        show_progress      = True
        )
    forecaster.fit(y=data['close'])
    pred_ultimo_valor = forecaster.predict(steps=steps)
    pred_ultimo_valor = pd.DataFrame(pred_ultimo_valor)
    predicciones_4h = pd.concat(objs=[predicciones,pred_ultimo_valor], axis=0)
    print("predicciones ",predicciones_4h)
    return predicciones_4h


def predict(db_path, lags=[1,30,90,180], steps=5):
    # Cargar y preparar los datos
    data = load_and_prepare_data(db_path)
    # Preprocesar el dataframe futuro
    # Crear, entrenar el modelo y realizar predicciones
    predicciones = create_and_train_model(data=data, lags=lags, steps=steps)
    return predicciones,data

def predict_sin_exog(db_path, lags=[1,30,90,180], steps=5,temporalidad=None):
    # Cargar y preparar los datos
    data = load_and_prepare_data_sin_exog(db_path,temporalidad=temporalidad)
    # Preprocesar el dataframe futuro
    # Crear, entrenar el modelo y realizar predicciones
    predicciones = create_train_model_sin_exog(data=data, lags=lags, steps=steps,temporalidad=temporalidad)
    return data, predicciones

if __name__ == "__main__":
    pass
    #data= load_and_prepare_data('Data/db/btc.db')
    #predictions = create_and_train_model(data)
    #print(predictions)
