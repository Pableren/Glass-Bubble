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
        return pd.read_sql_query(""" SELECT date, close, volume, volatility, rsi, ma_5, ma_20, ma_100 FROM btc_1d""", conn)
    elif temporalidad == '4H':
        return pd.read_sql_query("SELECT * FROM btc_4h", conn)
    elif temporalidad == '1H':
        return pd.read_sql_query("SELECT * FROM btc_1h", conn)
    elif temporalidad == '5M':
        return pd.read_sql_query("SELECT * FROM btc_5m", conn)

def entrenar_y_predecir(temporalidad):
    #data = func.load_and_prepare_data_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
    data, predictions = func.predict_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
    return data, predictions

def cortar_data(temporalidad):
    db.cortar_data(temporalidad=temporalidad)
    return None

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
    predicciones = predictions
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

def get_cell_style(value, threshold=30000):
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

#115f8f

# Función para generar la tabla con estilos en HTML
def create_info_table_with_style(volume, volatility):
    # Crear la tabla en HTML
    html = f"""
    <table style="width:100%; border-collapse: collapse; background-color: #397cc4;">
        <tr>
            <th style="border: 1px solid black; padding: 3px;">Variable</th>
            <th style="border: 1px solid black; padding: 3px;">Valor</th>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Volumen</td>
            <td style="border: 1px solid black; padding: 3px; {get_cell_style(volume)}">{volume}</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Volatilidad</td>
            <td style="border: 1px solid black; padding: 3px; {estilo_porcentuales(volatility)}">{volatility}</td>
        </tr>
    </table>
    """
    return html


#temporalidad_seleccionada = st.selectbox("Selecciona la temporalidad", lista_temporalidades)

# Crear columnas: la primera columna será el panel de información y la segunda columna los gráficos
col1, col2 = st.columns([1, 3])  # Ajusta las proporciones de ancho

# Cargar los datos según la temporalidad seleccionada
#data = cargar_datos(temporalidad_seleccionada)

#info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])

# Mostrar el panel de información en la columna izquierda
with col1:
    #temporalidad_seleccionada = st.selectbox("Selecciona la temporalidad", lista_temporalidades)
    temporalidad_seleccionada = st.selectbox("Selecciona la temporalidad", lista_temporalidades, index=0)
    st.subheader("Panel de Información")
    #info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])
    #st.table(info_panel_table)  # Mostrar tabla
    data = cargar_datos(temporalidad_seleccionada)
    if data is not None:
        #info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])
        #st.table(info_panel_table)  # Mostrar tabla con los datos cargados
        
        volume = data['volume'].iloc[-1]
        volatility = data['volatility'].iloc[-1]
        # Generar la tabla con estilo y mostrarla con HTML
        styled_table = create_info_table_with_style(volume, volatility)
        st.markdown(styled_table, unsafe_allow_html=True)
        
        
    
    if st.button('Actualizar Datos'):
        data = cargar_datos(temporalidad_seleccionada)
        #if data is not None:
        #    info_panel_table = create_info_table(data['volume'].iloc[-1], data['volatility'].iloc[-1])
        #    st.table(info_panel_table)  # Mostrar tabla con datos actualizados
        if temporalidad_seleccionada=='1D':
            db.actualizarData1d()
        elif temporalidad_seleccionada=='4H':
            db.actualizarData4h()
        elif temporalidad_seleccionada=='1H':
            db.actualizarData1h()
        elif temporalidad_seleccionada=='5M':
            db.actualizarData5m()
        data = cargar_datos(temporalidad_seleccionada)
    
    # Botón para entrenar el modelo
    if st.button('Entrenar Modelo'):
        data, predictions = entrenar_y_predecir(temporalidad_seleccionada)
        # Guardar las predicciones para usarlas en la segunda columna
        st.session_state['data'] = data
        st.session_state['predictions'] = predictions
    else:
        # Guardar los datos reales para usarlos en la segunda columna
        st.session_state['data'] = data
        st.session_state['predictions'] = None

    if st.button('Recortar datos'):
        recortar_data = cortar_data(temporalidad=temporalidad_seleccionada)


# Mostrar el gráfico en la columna derecha
with col2:
    st.subheader(f"Gráfico {temporalidad_seleccionada}")
    # Verificar si hay predicciones y mostrar el gráfico correspondiente
    if st.session_state.get('predictions') is not None:
        st.plotly_chart(crear_grafico_predicciones(st.session_state['data'], st.session_state['predictions'], temporalidad_seleccionada))
    else:
        st.plotly_chart(crear_grafico_valores_reales(st.session_state['data'], temporalidad_seleccionada))



