import pandas as pd
import sqlite3 as sql

from lightgbm import LGBMRegressor
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb # pip install xgboost
from sklearn.model_selection import TimeSeriesSplit
import ta# pip install TA-Lib
import joblib # pip install joblib

#df_1d = pd.read_parquet('Data/datasets/btc_1d.parquet')
conn = sql.connect('Data/db/btc.db')
cursor = conn.cursor()
df_1d = pd.read_sql_query("SELECT * FROM btc_1d", conn)
df_1d['date'] = pd.to_datetime(df_1d['date'])
#print(df_1d)
def crear_modelo():
    #tss = TimeSeriesSplit(n_splits=5,test_size=365,gap=2)
    reg = xgb.XGBRegressor(base_score=0.5,
                       booster='gbtree',    
                       n_estimators=500,
                       objective='reg:squarederror',
                       max_depth=3,
                       learning_rate=0.2)
    return reg

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    #print(df)
    #df.set_index(keys='date',inplace=True)
    if df.index.dtype == 'datetime64[ns]':
        pass
    else:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index(keys='date',inplace=True)
    #df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    df.reset_index(inplace=True)
    return df

def add_lags(df):
    df['lag1_target'] = df['close'].shift(120)
    df['lag2_target'] = df['close'].shift(180)
    df['lag3_target'] = df['close'].shift(360)
    
    return df

def entrenar_modelo_predicciones(df):
    df = create_features(df)
    df = add_lags(df)
    print(df.head())
    FEATURES = ['rsi_14_shifted','dayofyear', 'dayofweek',
                'quarter', 'month', 'year',
                'lag1_TR', 'lag2_TR', 'lag1_ATR','lag2_ATR',
                'lag1_target', 'lag2_target', 'lag3_target']
    TARGET = 'close'
    X_all = df[FEATURES]
    y_all = df[TARGET]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #modelo = LogisticRegression()
    modelo = crear_modelo()
    #modelo.fit(X_train, y_train)
    modelo.fit(X_all, y_all,
        eval_set=[(X_all, y_all)],
        verbose=100)
    fecha_maxima = df.index.max()
    fecha_maxima_timestamp = pd.Timestamp(fecha_maxima)
    mes = fecha_maxima_timestamp + pd.DateOffset(weeks=4)
    #un_mes_mas = fecha_maxima_timestamp + pd.DateOffset(month=1)
    future = pd.date_range(fecha_maxima, mes)
    future_df = pd.DataFrame(index=future)
    #print(future_df.index)
    future_df['isFuture'] = True
    df['isFuture'] = False
    df_and_future = pd.concat([df, future_df])
    df_and_future = create_features(df_and_future)
    df_and_future = add_lags(df_and_future)
    df_and_future.fillna(method='bfill', inplace=True)
    #print(df_and_future)
    window_14 = 14
    window_28 = 28
    df_and_future['rsi_14'] = ta.momentum.RSIIndicator(close=df_and_future['close'],window=window_14).rsi()
    df_and_future['rsi_28'] = ta.momentum.RSIIndicator(close=df_and_future['close'],window=window_28).rsi()
    shift_periods = 7  # Puedes cambiar esto al número de periodos que desees
    df_and_future['rsi_14_shifted'] = df_and_future['rsi_14'].shift(+shift_periods)
    shift_periods = 14
    df_and_future['rsi_28_shifted'] = df_and_future['rsi_28'].shift(+shift_periods)
    df_and_future['lag1_TR'] = df_and_future['TR'].shift(45)
    df_and_future['lag2_TR'] = df_and_future['TR'].shift(90)
    df_and_future['lag1_ATR'] = df_and_future['ATR'].shift(45)
    df_and_future['lag2_ATR'] = df_and_future['ATR'].shift(90)
    
    
    future_w_features = df_and_future.query('isFuture').copy()
    historical_w_features = df_and_future.query('~isFuture').copy()
    # Realizar predicciones sobre el último valor del dataframe original y los valores futuros
    historical_w_features['pred'] = modelo.predict(historical_w_features[FEATURES])
    future_w_features['pred'] = modelo.predict(future_w_features[FEATURES])
    all_predictions = pd.concat([historical_w_features[['pred']], future_w_features[['pred']]])
   #print(df_and_future[['close','rsi_14_shifted','lag1','dayofyear']])
    #print(future_w_features.shape)
    #future_w_features['pred'] = modelo.predict(future_w_features[FEATURES])
    #print(future_w_features[['rsi_14_shifted','rsi_28_shifted','lag1','close','dayofyear','pred']])
    return all_predictions,historical_w_features


#a,b = entrenar_modelo_predicciones(df=df_1d)
#print("valor de a",a) # future_w_features
#print("valor de b",b) # df_and_future
"""
def guardar_modelo(modelo, modelo_path):
    joblib.dump(modelo, modelo_path)
    
"""