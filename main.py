import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3 as sql
import Scripts.db_crud as db
from Scripts.train_save_model import entrenar_modelo_predicciones
#streamlit run main.py
# Conectar a la base de datos
conn = sql.connect('Data/db/btc.db')
cursor = conn.cursor()

# Cargar los datos
data_1d = pd.read_sql_query(""" SELECT date, close, volume, volatility, time, rsi_14, rsi_28, ma_5, ma_20, ma_100,
                                    MiddleBand, UpperBand, LowerBand, K, D, TR, ATR, TP, CCI, lag1_TR, lag2_TR,
                                    lag1_ATR, lag2_ATR
                            FROM btc_1d""", conn)
data_1d['date'] = pd.to_datetime(data_1d['date'])

data_4h = pd.read_sql_query("SELECT * FROM btc_4h", conn)
data_1h = pd.read_sql_query("SELECT * FROM btc_1h", conn)
data_5m = pd.read_sql_query("SELECT * FROM btc_5m", conn)

# Funciones para estilo de celdas (No son directamente aplicables en Streamlit pero pueden ser usadas para condicionales en visualizaciones)
def estilo_cci(value, lim_inf=-100, lim_sup=100):
    if value < lim_inf:
        return "background-color: green; color: black;"
    elif value > lim_sup:
        return "background-color: red; color: black;"
    else:
        return "background-color: grey; color: black;"

def estilo_porcentuales(value, lim_inf=20, lim_sup=80):
    if value < lim_inf:
        return "background-color: green; color: black;"
    elif value > lim_sup:
        return "background-color: red; color: black;"
    else:
        return "background-color: grey; color: black;"

def get_cell_style(value, threshold=35000):
    if value > threshold:
        return "background-color: green; color: white;"
    elif value < threshold:
        return "background-color: red; color: white;"
    return ""

# Crear tablas de información para mostrar
def create_info_table(volume, volatility):
    return pd.DataFrame({
        "Variable": ["Volumen", "Volatilidad"],
        "Valor": [volume, volatility]
    })

info_panel_table = create_info_table(data_1d['volume'].iloc[-1], data_1d['volatility'].iloc[-1])

# Visualización con Streamlit
st.title("Btc prediction with Streamlit")

# Botones para actualizar datos y entrenar modelo
if st.button('Actualizar Datos'):
    db.actualizarData1d()
    data_1d = pd.read_sql_query(""" SELECT date, close, volume, volatility, time, rsi_14, rsi_28, ma_5, ma_20, ma_100,
                                        MiddleBand, UpperBand, LowerBand, K, D, TR, ATR, TP, CCI, lag1_TR, lag2_TR,
                                        lag1_ATR, lag2_ATR
                                FROM btc_1d""", conn)
    data_4h = pd.read_sql_query("SELECT * FROM btc_4h", conn)
    data_1h = pd.read_sql_query("SELECT * FROM btc_1h", conn)
    data_5m = pd.read_sql_query("SELECT * FROM btc_5m", conn)

if st.button('Entrenar Modelo'):
    prediccion, actual = entrenar_modelo_predicciones(df=data_1d)
    datos_plot = pd.DataFrame({'actual': actual['close'], 'prediccion': prediccion['pred']})
    valores_graficos = pd.DataFrame({'ma_5': actual['ma_5'], 'ma_20': actual['ma_20'], 'ma_100': actual['ma_100']})
    valores_graficos_recortado = valores_graficos.tail(500)
    datos_plot_recortado = datos_plot.tail(500)
    df_concatenado = pd.concat([datos_plot_recortado, valores_graficos_recortado], ignore_index=False, join='outer')
    fig_1d = px.line(df_concatenado, x=df_concatenado.index, y=df_concatenado.columns, title='btc close vs predict')
else:
    # Gráficas iniciales
    fig_1d = px.line(data_1d, x='date', y=['close', 'ma_5', 'ma_20', 'ma_100'], title='btc close vs predictions 1d')

fig_4h = px.line(data_4h, x='date', y='close', title='btc close vs predictions 4h')
fig_1h = px.line(data_1h, x='date', y='close', title='btc close vs predictions 1h')
fig_5m = px.line(data_5m, x='date', y='close', title='btc close vs predictions 5m')

# Mostrar gráficos
st.plotly_chart(fig_1d)
st.plotly_chart(fig_4h)
st.plotly_chart(fig_1h)
st.plotly_chart(fig_5m)

# Mostrar tabla de información
st.subheader("Panel de Información")
st.table(info_panel_table)
