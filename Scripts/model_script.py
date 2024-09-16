
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

def load_and_prepare_data_sin_exog(db_path,temporalidad):
    conn = sql.connect(db_path)
    data = pd.read_sql_query(f"""SELECT date, close, volume, volatility, rsi, ma_5, ma_20, ma_100, CCI, K, D,
               MiddleBand, UpperBand, LowerBand FROM btc_{temporalidad}""", conn)
    conn.close()
    print("data en model_script",data.tail(3))
    return data

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
    print("metrica: ",metrica)
    return predicciones_4h

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
