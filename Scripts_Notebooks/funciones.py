import pandas as pd
import numpy as np
#from datetime import timedelta
import ta
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly_express as px
import seaborn as sns

def graficar_histogramas(df, columnas):
    # Definir el número de columnas
    n_cols = 3
    # Calcular el número de filas necesarias
    n_rows = int(np.ceil(len(columnas) / n_cols))
    
    # Crear una figura con subplots distribuidos en n_rows x n_cols
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5*n_rows))
    
    # Aplanar los ejes para facilitar el acceso con un solo índice
    axes = axes.flatten()

    for i, columna in enumerate(columnas):
        # Graficar histograma en su posición correspondiente
        sns.histplot(data=df, x=columna, ax=axes[i], bins=50)
        axes[i].set_xlim(auto=True)
        axes[i].set_ylim(auto=True)
    
    # Eliminar ejes vacíos si hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.show()



def calcular_recompensa_y_cuenta_regresiva_df4(df_4h, bloques_minados_hasta_hoy=740000):
    # Dict con la info de los halvings del Bitcoin
    #print(df_4h['date'].tail())
    #df_4h['date'] = pd.to_datetime(df_4h['date'])
    #print(df_4h['date'].tail())
    df_4h.set_index(keys='date',inplace=True)
    #print(df_4h.tail())
    btc_halving = {'halving': [0, 1, 2, 3, 4],
                   'date': ['2009-01-03 00:00:00', '2012-11-28 00:00:00', '2016-07-09 00:00:00', '2020-05-11 00:00:00', np.nan],
                   'reward': [50, 25, 12.5, 6.25, 3.125],
                   'halving_block_number': [0, 210000, 420000, 630000, 840000]
                  }

    # Cálculo siguiente halving
    bloques_por_dia = 144  # Aproximadamente 144 bloques por día

    # Calcula los bloques restantes para el próximo halving (se espera alrededor de 2028)
    bloques_restantes = 840000 - bloques_minados_hasta_hoy
    dias_restantes = bloques_restantes / bloques_por_dia

    # Fecha actual
    fecha_actual = pd.to_datetime('today').replace(microsecond=0, second=0, minute=0, hour=0)

    # Estima la fecha del próximo halving
    next_halving = fecha_actual + timedelta(days=dias_restantes)
    next_halving = next_halving.strftime('%Y-%m-%d')

    btc_halving['date'][-1] = next_halving

    print(f'El próximo halving ocurrirá aproximadamente el: {next_halving}')

    # Añadir columnas 'reward' y 'countdown_halving' al DataFrame
    df_4h['reward'] = np.nan
    df_4h['countdown_halving'] = np.nan

    for i in range(len(btc_halving['halving']) - 1):
        # Fecha inicial y final de cada halving
        if btc_halving['date'][i] < df_4h.index.min().strftime('%Y-%m-%d %H:%M:%S'):
            start_date = df_4h.index.min().strftime('%Y-%m-%d %H:%M:%S')
            print("start_date", start_date)
        else:
            start_date = btc_halving['date'][i]
            print("start_date", start_date)

        end_date = btc_halving['date'][i + 1]
        mask = (df_4h.index >= start_date) & (df_4h.index < end_date)

        # Rellenar columna 'reward' con las recompensas de minería
        df_4h.loc[mask, 'reward'] = btc_halving['reward'][i]

        # Rellenar columna 'countdown_halving' con los intervalos de 4 horas restantes
        time_to_next_halving = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        total_hours = time_to_next_halving.days * 24
        four_hour_intervals = total_hours // 4
        df_4h.loc[mask, 'countdown_halving'] = np.arange(four_hour_intervals)[::-1][:mask.sum()]

    # Considerar el próximo halving después del último registrado
    last_halving_date = pd.to_datetime(btc_halving['date'][-2])
    next_halving_date = pd.to_datetime(btc_halving['date'][-1])

    if df_4h.index.max() >= last_halving_date:
        mask = (df_4h.index >= last_halving_date) & (df_4h.index < next_halving_date)
        df_4h.loc[mask, 'reward'] = btc_halving['reward'][-1]

        time_to_next_halving = next_halving_date - last_halving_date
        total_hours = time_to_next_halving.days * 24
        four_hour_intervals = total_hours // 4
        df_4h.loc[mask, 'countdown_halving'] = np.arange(four_hour_intervals)[::-1][:mask.sum()]
    
    df_4h.reset_index(inplace=True)
    return df_4h


def calcular_recompensa_y_cuenta_regresiva_1d(df_1d, bloques_minados_hasta_hoy=740000):
    #if 'date' in df_1d.columns:
    df_1d['date'] = pd.to_datetime(df_1d['date'])
    df_1d.set_index(keys='date', inplace=True)
    #df_1d['date'] = pd.to_datetime(df_1d['date'])
    #df_1d.set_index(keys='date', inplace=True)

    btc_halving = {
        'halving': [0, 1, 2, 3, 4],
        'date': ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', np.nan],
        'reward': [50, 25, 12.5, 6.25, 3.125],
        'halving_block_number': [0, 210000, 420000, 630000, 840000]
    }

    bloques_por_dia = 144
    bloques_restantes = 840000 - bloques_minados_hasta_hoy
    dias_restantes = bloques_restantes / bloques_por_dia
    fecha_actual = pd.to_datetime('today').replace(microsecond=0, second=0, minute=0, hour=0)
    next_halving = fecha_actual + timedelta(days=dias_restantes)
    btc_halving['date'][-1] = next_halving.strftime('%Y-%m-%d %H:%M:%S')

    print(f'El próximo halving ocurrirá aproximadamente el: {btc_halving["date"][-1]}')

    df_1d['reward'] = np.nan
    df_1d['countdown_halving'] = np.nan

    for i in range(len(btc_halving['halving']) - 1):
        start_date = max(pd.to_datetime(btc_halving['date'][i]), df_1d.index.min())
        end_date = pd.to_datetime(btc_halving['date'][i + 1])
        mask = (df_1d.index >= start_date) & (df_1d.index < end_date)

        df_1d.loc[mask, 'reward'] = btc_halving['reward'][i]

        if mask.sum() > 0:
            dates_in_mask = pd.date_range(start=start_date, end=end_date, periods=mask.sum())
            countdown_values = (end_date - dates_in_mask).days
            df_1d.loc[mask, 'countdown_halving'] = countdown_values

    last_halving_date = pd.to_datetime(btc_halving['date'][-2])
    next_halving_date = pd.to_datetime(btc_halving['date'][-1])

    if df_1d.index.max() >= last_halving_date:
        mask = (df_1d.index >= last_halving_date) & (df_1d.index < next_halving_date)
        df_1d.loc[mask, 'reward'] = btc_halving['reward'][-1]

        if mask.sum() > 0:
            dates_in_mask = pd.date_range(start=last_halving_date, end=next_halving_date, periods=mask.sum())
            countdown_values = (next_halving_date - dates_in_mask).days
            df_1d.loc[mask, 'countdown_halving'] = countdown_values
    df_1d.reset_index(inplace=True)
    return df_1d





"""

def calcular_recompensa_y_cuenta_regresiva_1d(df_1d, bloques_minados_hasta_hoy=740000):
    # Dict con la info de los halvings del Bitcoin
    #print(df_1d['date'].tail())
    df_1d['date'] = pd.to_datetime(df_1d['date'])
    #print(df_1d['date'].tail())
    df_1d.set_index(keys='date',inplace=True)
    #print(df_1d['date'].tail())
    btc_halving = {'halving': [0, 1, 2, 3, 4],
                   'date': ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', np.nan],
                   'reward': [50, 25, 12.5, 6.25, 3.125],
                   'halving_block_number': [0, 210000, 420000, 630000, 840000]
                  }

    # Cálculo siguiente halving
    bloques_por_dia = 144  # Aproximadamente 144 bloques por día

    # Calcula los bloques restantes para el próximo halving (se espera alrededor de 2028)
    bloques_restantes = 840000 - bloques_minados_hasta_hoy
    dias_restantes = bloques_restantes / bloques_por_dia

    # Fecha actual
    fecha_actual = pd.to_datetime('today').replace(microsecond=0, second=0, minute=0, hour=0)

    # Estima la fecha del próximo halving
    next_halving = fecha_actual + timedelta(days=dias_restantes)
    next_halving = next_halving.strftime('%Y-%m-%d %H:%M:%S')

    btc_halving['date'][-1] = next_halving

    print(f'El próximo halving ocurrirá aproximadamente el: {next_halving}')

    # Añadir columnas 'reward' y 'countdown_halving' al DataFrame
    df_1d['reward'] = np.nan
    df_1d['countdown_halving'] = np.nan

    for i in range(len(btc_halving['halving']) - 1):
        # Fecha inicial y final de cada halving
        if btc_halving['date'][i] < df_1d.index.min().strftime('%Y-%m-%d %H:%M:%S'):
            start_date = df_1d.index.min().strftime('%Y-%m-%d %H:%M:%S')
            print("start_date", start_date)
        else:
            start_date = btc_halving['date'][i]
            print("start_date", start_date)
        
        end_date = btc_halving['date'][i + 1]
        mask = (df_1d.index >= start_date) & (df_1d.index < end_date)
        
        # Rellenar columna 'reward' con las recompensas de minería
        df_1d.loc[mask, 'reward'] = btc_halving['reward'][i]
        
        # Rellenar columna 'countdown_halving' con los días restantes
        time_to_next_halving = pd.to_datetime(end_date) - pd.to_datetime(start_date)
        print("masacara: ",mask)
        print("siguiente halving: ",time_to_next_halving)
        df_1d.loc[mask, 'countdown_halving'] = np.arange(time_to_next_halving.days)[::-1][:mask.sum()]

    # Considerar el próximo halving después del último registrado
    last_halving_date = pd.to_datetime(btc_halving['date'][-2])
    next_halving_date = pd.to_datetime(btc_halving['date'][-1])

    if df_1d.index.max() >= last_halving_date:
        mask = (df_1d.index >= last_halving_date) & (df_1d.index < next_halving_date)
        df_1d.loc[mask, 'reward'] = btc_halving['reward'][-1]
        
        time_to_next_halving = next_halving_date - last_halving_date
        df_1d.loc[mask, 'countdown_halving'] = np.arange(time_to_next_halving.days)[::-1][:mask.sum()]

    return df_1d
"""



def etl_1h_5m(df):
    df['return'] = ((df['close'] - df['open']) / df['open'])*100
    df['diff'] = df['close'] - df['open']
    df['volatility'] = df['high'] - df['low']
    window_14 = 14
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'],window=window_14).rsi()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()

    # Bandas de Bollinger solo se usara para graficar
    df['MiddleBand'] = df['close'].rolling(window=20).mean()
    df['UpperBand'] = df['MiddleBand'] + 2*df['close'].rolling(window=20).std()
    df['LowerBand'] = df['MiddleBand'] - 2*df['close'].rolling(window=20).std()

    #Oscilador Estocástico
    df['K'] = 100 * ((df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()))
    df['D'] = df['K'].rolling(window=3).mean()
    #Índice de Canal de Materias Primas (CCI)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(window=20).mean()) / (0.015 * df['TP'].rolling(window=20).std())    
    
    last_timestamp = df['time'].iloc[-1]
    last_timestamp = float(last_timestamp)
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    #df['date'] = df['time'].apply(lambda x: timedelta(milliseconds=(last_timestamp - x)))
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


def etl_1d_4h(df):
    df['return'] = ((df['close'] - df['open']) / df['open'])*100
    df['diff'] = df['close'] - df['open']
    df['volatility'] = df['high'] - df['low']
    window_14 = 14
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'],window=window_14).rsi()
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_100'] = df['close'].rolling(window=100).mean()
    # Bandas de Bollinger solo se usara para graficar
    df['MiddleBand'] = df['close'].rolling(window=20).mean()
    df['UpperBand'] = df['MiddleBand'] + 2*df['close'].rolling(window=20).std()
    df['LowerBand'] = df['MiddleBand'] - 2*df['close'].rolling(window=20).std()
    #Oscilador Estocástico
    df['K'] = 100 * ((df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()))
    df['D'] = df['K'].rolling(window=3).mean()

    #Índice de Canal de Materias Primas (CCI)
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['CCI'] = (df['TP'] - df['TP'].rolling(window=20).mean()) / (0.015 * df['TP'].rolling(window=20).std())
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    #df['date'] = pd.to_datetime(df['time'], unit='ms').dt.normalize()
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df


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


def calcular_recompensa_y_cuenta_regresiva_1d_future(df_1d, bloques_minados_hasta_hoy=740000):
    btc_halving = {
        'halving': [0, 1, 2, 3, 4],
        'date': ['2009-01-03', '2012-11-28', '2016-07-09', '2020-05-11', np.nan],
        'reward': [50, 25, 12.5, 6.25, 3.125],
        'halving_block_number': [0, 210000, 420000, 630000, 840000]
    }

    bloques_por_dia = 144
    bloques_restantes = 840000 - bloques_minados_hasta_hoy
    dias_restantes = bloques_restantes / bloques_por_dia
    fecha_actual = pd.to_datetime('today').replace(microsecond=0, second=0, minute=0, hour=0)
    next_halving = fecha_actual + timedelta(days=dias_restantes)
    btc_halving['date'][-1] = next_halving.strftime('%Y-%m-%d %H:%M:%S')

    print(f'El próximo halving ocurrirá aproximadamente el: {btc_halving["date"][-1]}')

    df_1d['reward'] = np.nan
    df_1d['countdown_halving'] = np.nan

    for i in range(len(btc_halving['halving']) - 1):
        start_date = max(pd.to_datetime(btc_halving['date'][i]), df_1d.index.min())
        end_date = pd.to_datetime(btc_halving['date'][i + 1])
        mask = (df_1d.index >= start_date) & (df_1d.index < end_date)

        df_1d.loc[mask, 'reward'] = btc_halving['reward'][i]

        if mask.sum() > 0:
            dates_in_mask = pd.date_range(start=start_date, end=end_date, periods=mask.sum())
            countdown_values = (end_date - dates_in_mask).days
            df_1d.loc[mask, 'countdown_halving'] = countdown_values

    last_halving_date = pd.to_datetime(btc_halving['date'][-2])
    next_halving_date = pd.to_datetime(btc_halving['date'][-1])

    if df_1d.index.max() >= last_halving_date:
        mask = (df_1d.index >= last_halving_date) & (df_1d.index < next_halving_date)
        df_1d.loc[mask, 'reward'] = btc_halving['reward'][-1]

        if mask.sum() > 0:
            dates_in_mask = pd.date_range(start=last_halving_date, end=next_halving_date, periods=mask.sum())
            countdown_values = (next_halving_date - dates_in_mask).days
            df_1d.loc[mask, 'countdown_halving'] = countdown_values
    #df_1d.reset_index(inplace=True)
    return df_1d
