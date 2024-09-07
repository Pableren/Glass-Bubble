import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3 as sql
import Scripts.db_crud as db
import Scripts.model_script as func
import plotly.graph_objects as go
#streamlit run main.py
# Conectar a la base de datos
conn = sql.connect('Data/db/btc.db')
cursor = conn.cursor()

# Cargar los datos para cada temporalidad
def cargar_datos(temporalidad):
    if temporalidad == '1D':
        return pd.read_sql_query(""" SELECT date, close, volume, volatility, rsi_14, ma_5, ma_20, ma_100 FROM btc_1d""", conn)
    elif temporalidad == '4H':
        return pd.read_sql_query("SELECT * FROM btc_4h", conn)
    elif temporalidad == '1H':
        return pd.read_sql_query("SELECT * FROM btc_1h", conn)
    elif temporalidad == '5M':
        return pd.read_sql_query("SELECT * FROM btc_5m", conn)

def entrenar_y_predecir(temporalidad):
    if temporalidad == '1D':
        data = func.load_and_prepare_data('Data/db/btc.db')
        predictions = func.create_and_train_model(data)
    else:
        data = func.load_and_prepare_data_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
        predictions = func.predict_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
    
    return data, predictions#

# Crear gráfico de valores reales
def crear_grafico_valores_reales(data, temporalidad):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], 
                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
    fig.update_layout(
        title=f'Valores Reales {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white'
    )
    return fig

# Crear gráfico de predicciones
def crear_grafico_predicciones(data, predictions, temporalidad):
    fig = go.Figure()
    predicciones = predictions[1]
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], 
                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
    fig.add_trace(go.Scatter(x=predicciones.index, y=predicciones.pred.values, 
                             mode='lines+markers', name='Predicciones en 5 Pasos', marker=dict(color='green')))
    
    fig.update_layout(
        title=f'Predicciones {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white'
    )
    return fig
st.set_page_config(layout="wide")  # Esto asegura que el ancho completo se use
st.title("Btc prediction with Streamlit")
lista_temporalidades = ['1D', '4H', '1H', '5M']
graficas = {}
graficas_predicciones = {}

# Crear placeholders para los gráficos de cada temporalidad
graficas_placeholders = {}
for temporalidad in lista_temporalidades:
    graficas_placeholders[temporalidad] = st.empty()  # Crear un espacio para cada gráfico

# Mostrar los gráficos iniciales con valores reales
for temporalidad in lista_temporalidades:
    data = cargar_datos(temporalidad)
    graficas_placeholders[temporalidad].plotly_chart(crear_grafico_valores_reales(data, temporalidad))

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




temporalidad_seleccionada = st.selectbox("Selecciona la temporalidad", lista_temporalidades)

# Crear columnas: la primera columna será el panel de información y la segunda columna los gráficos
col1, col2 = st.columns([1, 3])  # Ajusta las proporciones de ancho

# Cargar los datos según la temporalidad seleccionada
data = cargar_datos(temporalidad_seleccionada)

info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])


# Mostrar el panel de información en la columna izquierda
with col1:
    st.subheader("Panel de Información")
    
    # Cargar los datos para la tabla de información
    data = cargar_datos(temporalidad_seleccionada)
    info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])
    st.table(info_panel_table)  # Mostrar tabla
    
    # Colocar botones debajo del panel de información
    if st.button('Actualizar Datos'):
        if temporalidad_seleccionada=='1D':
            db.actualizarData1d()
        elif temporalidad_seleccionada=='4H':
            db.actualizarData4h()
        elif temporalidad_seleccionada=='1H':
            db.actualizarData1h()
        elif temporalidad_seleccionada=='5M':
            db.actualizarData5m()
        data = cargar_datos(temporalidad_seleccionada)
    
    if st.button('Entrenar Modelo'):
        data, predictions = entrenar_y_predecir(temporalidad_seleccionada)

# Mostrar gráficos en la columna derecha
with col2:
    st.subheader(f"Gráficos {temporalidad_seleccionada}")
    st.plotly_chart(crear_grafico_valores_reales(data, temporalidad_seleccionada))


# Mostrar el panel de información en la columna izquierda
#with col1:
#    st.subheader("Panel de Información")
#    st.table(info_panel_table)
#
## Mostrar gráficos en la columna derecha
#with col2:
#    st.subheader(f"Gráficos {temporalidad_seleccionada}")
#    st.plotly_chart(crear_grafico_valores_reales(data, temporalidad_seleccionada))
#
#    # Actualizar gráficos al presionar los botones
#    if st.button('Actualizar Datos'):
#        if temporalidad_seleccionada == '1D':
#            db.actualizarData1d()
#        elif temporalidad_seleccionada == '4H':
#            db.actualizarData4h()
#        elif temporalidad_seleccionada == '1H':
#            db.actualizarData1h()
#        elif temporalidad_seleccionada == '5M':
#            db.actualizarData5m()
#        # Volver a cargar los datos después de la actualización
#        data = cargar_datos(temporalidad_seleccionada)
#        # Actualizar el gráfico con los datos crudos
#        graficas_placeholders[temporalidad_seleccionada].plotly_chart(crear_grafico_valores_reales(data, temporalidad_seleccionada))
#
#    # Botón para entrenar modelo
#    
## Botón para entrenar modelo
#    if st.button('Entrenar Modelo'):
#        data, predictions = entrenar_y_predecir(temporalidad_seleccionada)
#        # Reemplazar el gráfico original de la temporalidad seleccionada con el nuevo gráfico de predicciones
#        graficas_placeholders[temporalidad_seleccionada].plotly_chart(crear_grafico_predicciones(data, predictions, temporalidad_seleccionada))
#
#
#        #graficas[temporalidad_seleccionada] = crear_grafico_predicciones(data, predictions, temporalidad_seleccionada)
#        #st.plotly_chart(graficas[temporalidad_seleccionada])



# Botón para actualizar datos según la temporalidad
# Actualizar gráficos al presionar los botones
#if st.button('Actualizar Datos'):
#    if temporalidad_seleccionada == '1D':
#        db.actualizarData1d()
#    elif temporalidad_seleccionada == '4H':
#        db.actualizarData4h()
#    elif temporalidad_seleccionada == '1H':
#        db.actualizarData1h()
#    elif temporalidad_seleccionada == '5M':
#        db.actualizarData5m()
#    # Volver a cargar los datos después de la actualización
#    data = cargar_datos(temporalidad_seleccionada)
#    # Actualizar el gráfico con los datos crudos
#    graficas_placeholders[temporalidad_seleccionada].plotly_chart(crear_grafico_valores_reales(data, temporalidad_seleccionada))#
#

## Botón para entrenar modelo
#if st.button('Entrenar Modelo'):
#    data, predictions = entrenar_y_predecir(temporalidad_seleccionada)
#    # Reemplazar el gráfico original de la temporalidad seleccionada con el nuevo gráfico de predicciones
#    graficas_placeholders[temporalidad_seleccionada].plotly_chart(crear_grafico_predicciones(data, predictions, temporalidad_seleccionada))
#    
#    
#    #graficas[temporalidad_seleccionada] = crear_grafico_predicciones(data, predictions, temporalidad_seleccionada)
#    #st.plotly_chart(graficas[temporalidad_seleccionada])




# Mostrar gráficos con predicciones si ya se entrenaron
if graficas_predicciones:
    st.subheader("Predicciones de Modelos")
    for temp, grafico in graficas_predicciones.items():
        st.plotly_chart(grafico)

# Mostrar tabla de información
#st.subheader("Panel de Información")
#st.table(info_panel_table)








# Cargar los datos
#data_1d = pd.read_sql_query(""" SELECT date, close, volume, volatility, time, rsi_14, rsi_28, ma_5, ma_20, ma_100,
#                                    MiddleBand, UpperBand, LowerBand, K, D, TR, ATR, TP, CCI, lag1_TR, lag2_TR,
#                                    lag1_ATR, lag2_ATR
#                            FROM btc_1d""", conn)
#data_1d['date'] = pd.to_datetime(data_1d['date'])
#
#data_4h = pd.read_sql_query("SELECT * FROM btc_4h", conn)
#data_1h = pd.read_sql_query("SELECT * FROM btc_1h", conn)
#data_5m = pd.read_sql_query("SELECT * FROM btc_5m", conn)


#fig_4h = px.line(data_4h, x='date', y='close', title='btc close vs predictions 4h')
#fig_1h = px.line(data_1h, x='date', y='close', title='btc close vs predictions 1h')
#fig_5m = px.line(data_5m, x='date', y='close', title='btc close vs predictions 5m')


# Botones para actualizar datos y entrenar modelo
#if st.button('Actualizar Datos'):
#    db.actualizarData1d()
#    data_1d = pd.read_sql_query(""" SELECT date, close, volume, volatility, time, rsi_14, rsi_28, ma_5, ma_20, ma_100,
#                                        MiddleBand, UpperBand, LowerBand, K, D, TR, ATR, TP, CCI, lag1_TR, lag2_TR,
#                                        lag1_ATR, lag2_ATR
#                                FROM btc_1d""", conn)
#    data_4h = pd.read_sql_query("SELECT * FROM btc_4h", conn)
#    data_1h = pd.read_sql_query("SELECT * FROM btc_1h", conn)
#    data_5m = pd.read_sql_query("SELECT * FROM btc_5m", conn)
#if st.button('Entrenar Modelo'):
#    #prediccion, actual = func.predict(db_path='Data/db/btc.db')
#    #data= func.load_and_prepare_data('Data/db/btc.db')
#    #predictions = func.create_and_train_model(data)
#    if temporalidad_seleccionada == '1D':
#        data= func.load_and_prepare_data('Data/db/btc.db')
#        predictions = func.create_and_train_model(data)
#    elif temporalidad_seleccionada == '4H':
#        #data= func.load_and_prepare_data()
#        data = func.load_and_prepare_data_sin_exog('Data/db/btc.db',temporalidad='4H')
#        predictions = func.predict_sin_exog(db_path='Data/db/btc.db',temporalidad='4H')
#    elif temporalidad_seleccionada == '1H':
#        data = func.load_and_prepare_data_sin_exog('Data/db/btc.db',temporalidad='1H')
#        predictions = func.predict_sin_exog(db_path='Data/db/btc.db',temporalidad='1H')
#    elif temporalidad_seleccionada == '5M':
#        data = func.load_and_prepare_data_sin_exog('Data/db/btc.db',temporalidad='5M')
#        predictions = func.predict_sin_exog(db_path='Data/db/btc.db',temporalidad='5M')
#    print(predictions)
#    #ultimos_valores_close = data['close'].iloc[-100:]
#    fig = go.Figure()
#    # Agregar los valores reales (últimos 10 valores)
#    fig.add_trace(go.Scatter(x=data.index[-500:], y=data.close.values[-500:], 
#                         mode='lines+markers', 
#                         name='Valores Reales', 
#                         marker=dict(color='blue', symbol='cross')))
#    fig.add_trace(go.Scatter(x=predictions.index, y=predictions.values, 
#                         mode='lines+markers', 
#                         name='Predicciones en 5 Pasos', 
#                         marker=dict(color='green')))
#    #datos_plot = pd.DataFrame({'actual': data['close'], 'prediccion': predictions.values})
#    #valores_graficos = pd.DataFrame({'ma_5': data['ma_5'], 'ma_20': data['ma_20'], 'ma_100': data['ma_100']})
#    
#    # Agregar los valores reales (últimos 10 valores)
#    fig.add_trace(go.Scatter(x=data.index[-100:], y=data.close.values[-100:], 
#                             name='Valores Reales'))
#    # Agregar las predicciones sobre los últimos valores y las predicciones futuras
#    fig.add_trace(go.Scatter(x=predictions.index[-100:], y=predictions.pred[-100:], 
#                             name='Valores Reales y Predicciones en Últimos 5 Pasos'))
#
#
#    
#    fig.update_layout(
#        title='Predicciones',
#        xaxis_title='Fecha',
#        yaxis_title='Precio',
#        legend=dict(x=0, y=1.1),
#        margin=dict(l=40, r=40, t=40, b=40),
#        hovermode='x',
#        paper_bgcolor='dodgerblue',
#        plot_bgcolor='deepskyblue',
#        template='plotly_white')
#    #fig.show()
#    #st.plotly_chart(fig)
#    #valores_graficos_recortado = valores_graficos.tail(500)
#    #datos_plot_recortado = datos_plot.tail(500)
#    #df_concatenado = pd.concat([datos_plot_recortado, valores_graficos_recortado], ignore_index=False, join='outer')
#    #fig_1d = px.line(df_concatenado, x=df_concatenado.index, y=df_concatenado.columns, title='btc close vs predict')
#else:
#    # Gráficas iniciales
#    fig_1d = px.line(data_1d, x='date', y=['close', 'ma_5', 'ma_20', 'ma_100'], title='btc close vs predictions 1d')



# Mostrar gráficos
#st.plotly_chart(fig_1d)
#st.plotly_chart(fig_4h)
#st.plotly_chart(fig_1h)
#st.plotly_chart(fig_5m)




# Entrenar modelo y obtener predicciones
#def entrenar_y_predecir(temporalidad):
#    if temporalidad == '1D':
#        data = func.load_and_prepare_data('Data/db/btc.db')
#        predictions = func.create_and_train_model(data)
#    else:
#        data = func.load_and_prepare_data_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
#        predictions = func.predict_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
#    
#    return data, predictions#

## Crear gráfico de predicciones
#def crear_grafico(data, predictions, temporalidad):
#    fig = go.Figure()
#    predicciones = predictions[1]
#    #print(predicciones)
#    #print(predicciones.index)
#    #print(data)
#    #print(data.index)
#    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data.close.values[-200:], 
#                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
#    fig.add_trace(go.Scatter(x=predicciones.index, y=predicciones.pred.values, 
#                             mode='lines+markers', name='Predicciones en 5 Pasos', marker=dict(color='green')))
#    
#    fig.update_layout(
#        title=f'Predicciones {temporalidad}',
#        xaxis_title='Fecha',
#        yaxis_title='Precio',
#        hovermode='x',
#        template='plotly_white'
#    )
#    return fig