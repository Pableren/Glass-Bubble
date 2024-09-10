import sqlite3 as sql
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
import ta
#import funciones
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import funciones
#from funciones import 
#from Scripts import *
def createDB():
    conn = sql.connect('Data/db/btc.db') # objeeto de la clase coneccion
    conn.commit()
    conn.close()
    
def eliminarTabla(tabla):
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    tabla = str(tabla)
    consulta = f"DROP TABLE {tabla};"
    cursor.execute(consulta)
    conn.commit()
    conn.close()
    
def crear_tabla(table_name):
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    consulta = f"""
        CREATE TABLE {table_name} (
            date TEXT,time TEXT,open REAL,high REAL,low REAL,close REAL,volume REAL,
            return REAL,diff REAL,volatility REAL,rsi REAL,ma_5 REAL,ma_20 REAL,ma_100 REAL,
            MiddleBand REAL, UpperBand REAL, LowerBand REAL,
            K REAL, D REAL, TP REAL, CCI REAL
        )
        """
    cursor.execute(consulta)
    conn.commit()
    conn.close()
    return None

def tables_create(table_list=['btc_1d','btc_4h','btc_1h','btc_5m']):
    for x in table_list:
        crear_tabla(x)
    return None

def tables_drop(table_list=['btc_1d','btc_4h','btc_1h','btc_5m']):
    for x in table_list:
        eliminarTabla(tabla=x)
    return None

# Metodo to_sql
def insertRows(tabla):
    tabla = str(tabla)
    df = pd.read_parquet(f'Data/datasets/{tabla}.parquet',engine='pyarrow') # MODIFICAR NOMBRES
    #df = pd.read_parquet('btc_1d.parquet',engine='pyarrow')
    print(df)
    conn = sql.connect('Data/db/btc.db')
    df.to_sql(f'{tabla}', conn, if_exists='append', index=False)
    #df.to_sql('btc_1d', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

def tables_fill(table_list=['btc_1d','btc_4h','btc_1h','btc_5m']):
    for x in table_list:
        insertRows(tabla=x)
    return None



    
def lecturaUltimoElemento():
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT * FROM btc_5m ORDER BY date DESC LIMIT 1;
                   """)
    valor = cursor.fetchall()
    print(valor)
    conn.commit()
    conn.close()

    
exchange = ccxt.binance()
def fetch_ohlcv_data(exchange, symbol, timeframe, since=None,limit=360):
    """
    Función para obtener datos OHLCV del exchange.
    Devuelve: DataFrame: DataFrame con datos OHLCV.
    """
    data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    #df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df




def actualizarData1h():
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT * FROM btc_1h ORDER BY time DESC LIMIT 1;
                   """)
    fila = cursor.fetchone()
    #date_old = fila[0]
    if fila is None:
        # No hay datos en la base de datos
        date_old = datetime(1970, 1, 1)
    else:
        date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
    date_new = datetime.now().replace(microsecond=0)# Obtener la fecha y hora actual
    diferencia_minutos = (date_new - date_old).total_seconds() / 60
    fifteen_dats_ms = 15*24 * 60 *60 * 1000
    #since = date_old
    #print(date_old)
    since = int(date_old.timestamp() * 1000)
    #since = int(date_old.timestamp() * 1000)# Calcular la cantidad de datos a obtener en base a la diferencia en minutos
    while diferencia_minutos > 60:
        datos_1h = fetch_ohlcv_data(exchange, 'BTC/USDT', '1h', since=since, limit=360)
        if datos_1h.empty:
            conn.close()
            return
        #mi valores de head, se concatenan con mis ultimos valores de la base
        datos_1h = funciones.etl_1h_5m(datos_1h)
        # Concatenar los nuevos datos y guardar en la base de datos
        df_actual = pd.read_sql_query("SELECT * FROM btc_1h", conn)
        datos_faltantes = datos_1h[~datos_1h['time'].isin(df_actual['time'])]
        if not datos_faltantes.empty:
            cursor.execute("""SELECT COUNT(*) FROM btc_1h;""")
            cantidad_registros_ingresados_viejos = cursor.fetchone()[0]
            datos_faltantes.to_sql('btc_1h', conn, if_exists='append', index=False)
            conn.commit()
            cursor.execute("""SELECT COUNT(*) FROM btc_1h;""")
            cantidad_registros_nuevos = cursor.fetchone()[0]
            print("cantidad de registros ingresados:",cantidad_registros_nuevos-cantidad_registros_ingresados_viejos)
            # Verificar la última fila después de la inserción
            cursor.execute("""SELECT * FROM btc_1h ORDER BY time DESC LIMIT 1;""")
            fila = cursor.fetchone()
            date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
            diferencia_minutos = (date_new-date_old).total_seconds()/60
        else:
            break
        since += fifteen_dats_ms
        #since = int(date_old.timestamp() * 1000)
        date_new = datetime.now().replace(microsecond=0)
        diferencia_minutos = (date_new - date_old).total_seconds() / 60
    df_actual = pd.read_sql_query("SELECT * FROM btc_1h", conn)
    df_actual = funciones.etl_1h_5m(df_actual)
    df_actual.to_sql('btc_1h', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()


def actualizarData4h():
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT * FROM btc_4h ORDER BY time DESC LIMIT 1;
                   """)
    fila = cursor.fetchone()
    if fila is None:
        date_old = datetime(1970, 1, 1)
    else:
        date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
    date_new = datetime.now().replace(microsecond=0)  # Obtener la fecha y hora actual
    diferencia_minutos = (date_new - date_old).total_seconds() / 60
    two_months = 60*24 * 60 *60 * 1000
    since = int(date_old.timestamp() * 1000)
    while diferencia_minutos > 240:  # 240 minutos = 4 horas
        datos_4h = fetch_ohlcv_data(exchange, 'BTC/USDT', '4h', since=since, limit=360)
        if datos_4h.empty:
            conn.close()
            return
        # Procesar los datos
        datos_4h = funciones.etl_1d_4h(datos_4h)
        # Concatenar los nuevos datos y guardar en la base de datos
        df_actual = pd.read_sql_query("SELECT * FROM btc_4h", conn)
        datos_faltantes = datos_4h[~datos_4h['time'].isin(df_actual['time'])]
        if not datos_faltantes.empty:
            cursor.execute("""SELECT COUNT(*) FROM btc_4h;""")
            cantidad_registros_ingresados_viejos = cursor.fetchone()[0]
            datos_faltantes.to_sql('btc_4h', conn, if_exists='append', index=False)
            conn.commit()
            cursor.execute("""SELECT COUNT(*) FROM btc_4h;""")
            cantidad_registros_nuevos = cursor.fetchone()[0]
            cursor.execute("""SELECT * FROM btc_4h ORDER BY time DESC LIMIT 1;""")
            fila = cursor.fetchone()
            date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
            diferencia_minutos = (date_new - date_old).total_seconds() / 60
        else:
            print("No hay valores faltantes.")
            break
        print("cantidad de registros ingresados:", cantidad_registros_nuevos - cantidad_registros_ingresados_viejos)
        since += two_months
        diferencia_minutos = (date_new - date_old).total_seconds() / 60
    df_actual = pd.read_sql_query("SELECT * FROM btc_4h", conn)
    df_actual = funciones.etl_1d_4h(df_actual)
    #df_actual = funciones.calcular_recompensa_y_cuenta_regresiva_df4(df_4h=df_actual)
    df_actual.to_sql('btc_4h', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    

def actualizarData1d():
    """
    Funcion para actualizar los datos de velas diarias
    """
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT * FROM btc_1d ORDER BY date DESC LIMIT 1;
                   """)
    fila = cursor.fetchone()
    if fila is None:
        date_old = datetime(1970, 1, 1)
    else:
        date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
    date_new = datetime.now()
    diferencia_minutos = (date_new - date_old).total_seconds() / 60
    one_year = 365 *24 * 60 *60 * 1000
    since = int(date_old.timestamp() * 1000)
    while diferencia_minutos > 1440:
        datos_1d = fetch_ohlcv_data(exchange, 'BTC/USDT', '1d', since=since, limit=365)
        if datos_1d.empty:
            conn.close()
            return
        datos_1d = funciones.etl_1d_4h(datos_1d)
        df_actual = pd.read_sql_query("SELECT * FROM btc_1d", conn)
        datos_faltantes = datos_1d[~datos_1d['time'].isin(df_actual['time'])]
        #print(datos_faltantes)
        if not datos_faltantes.empty:
            cursor.execute("""SELECT COUNT(*) FROM btc_1d;""")
            cantidad_registros_ingresados_viejos = cursor.fetchone()[0]
            datos_faltantes.to_sql('btc_1d', conn, if_exists='append', index=False)
            conn.commit()
            cursor.execute("""SELECT COUNT(*) FROM btc_1d;""")
            cantidad_registros_nuevos = cursor.fetchone()[0]
            cursor.execute("""SELECT * FROM btc_1d ORDER BY time DESC LIMIT 1;""")
            fila = cursor.fetchone()
            date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
        else:
            break
        print("cantidad de registros ingresados:", cantidad_registros_nuevos - cantidad_registros_ingresados_viejos)
        since += one_year
        diferencia_minutos = (date_new - date_old).total_seconds() / 60
    df_actual = pd.read_sql_query("SELECT * FROM btc_1d", conn)
    df_actual = funciones.etl_1d_4h(df_actual)
    df_actual = funciones.calcular_recompensa_y_cuenta_regresiva_1d(df_1d=df_actual)
    df_actual.to_sql('btc_1d', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
        


def actualizarData5m():
    conn = sql.connect('Data/db/btc.db')
    cursor = conn.cursor()
    cursor.execute("""
                   SELECT * FROM btc_5m ORDER BY time DESC LIMIT 1;
                   """)
    fila = cursor.fetchone()
    if fila is None:
        date_old = datetime(1970, 1, 1)
    else:
        date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
    date_new = datetime.now().replace(microsecond=0)  # Obtener la fecha y hora actual
    diferencia_minutos = (date_new - date_old).total_seconds() / 60
    #two_months = 60*24 * 60 *60 * 1000
    one_day = 24 * 60 *60 * 1000
    since = int(date_old.timestamp() * 1000)
    while diferencia_minutos > 5:
        datos_5m = fetch_ohlcv_data(exchange, 'BTC/USDT', '5m', since=since, limit=288)
        if datos_5m.empty:
            conn.close()
            return
        datos_5m = funciones.etl_1h_5m(datos_5m)
        df_actual = pd.read_sql_query("SELECT * FROM btc_5m", conn)
        datos_faltantes = datos_5m[~datos_5m['time'].isin(df_actual['time'])]
        if not datos_faltantes.empty:
            cursor.execute("""SELECT COUNT(*) FROM btc_5m;""")
            cantidad_registros_ingresados_viejos = cursor.fetchone()[0]
            datos_faltantes.to_sql('btc_5m', conn, if_exists='append', index=False)
            conn.commit()
            cursor.execute("""SELECT COUNT(*) FROM btc_5m;""")
            cantidad_registros_nuevos = cursor.fetchone()[0]
            cursor.execute("""SELECT * FROM btc_5m ORDER BY time DESC LIMIT 1;""")
            fila = cursor.fetchone()
            date_old = datetime.strptime(fila[0],"%Y-%m-%d %H:%M:%S")
            diferencia_minutos = (date_new - date_old).total_seconds() / 60
        else:
            print("No hay valores faltantes.")
            break
        print("cantidad de registros ingresados:", cantidad_registros_nuevos - cantidad_registros_ingresados_viejos)
        since += one_day
        diferencia_minutos = (date_new - date_old).total_seconds() / 60
    
    df_actual = pd.read_sql_query("SELECT * FROM btc_5m", conn)
    df_actual = funciones.etl_1h_5m(df_actual)
    df_actual.to_sql('btc_5m', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()
    
    

def cortar_data(temporalidad):
    """
    Funcion para almacenar los ultimos datos del bitcoin
    """
    table_name = 'btc'+'_'+temporalidad.lower()
    conn = sql.connect('Data/db/btc.db')
    df_actual = pd.read_sql_query(f"SELECT * FROM {table_name}",conn)
    total_data = 15000
    df_actual = df_actual[-total_data:]
    df_actual.to_sql(f'{table_name}',conn, if_exists='replace',index=False)
    conn.commit()
    conn.close()
    return None

if __name__ == "__main__":
    #tables_drop()
    #tables_create()
    #tables_fill()

    #actualizarData1h() #correcta
    #actualizarData4h() # Correcta
    #actualizarData1d() # correcta
    #actualizarData5m() # correcta 
    pass
