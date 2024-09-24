import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3 as sql
import Scripts.db_crud as db
import Scripts.model_script as f
import base64  # Para codificar la imagen a base64
import plotly.graph_objects as go
#streamlit run main.py
# Conectar a la base de datos
conn = sql.connect('Data/db/btc.db')
cursor = conn.cursor()
# Inicializar session_state para datos y predicciones si no existen
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'temporalidad_seleccionada' not in st.session_state:
    st.session_state.temporalidad_seleccionada = '1D'  # Temporalidad por defecto
if 'actualizar_panel' not in st.session_state:
    st.session_state.actualizar_panel = False  # Flag para actualizar el panel

#AGG
# Función para actualizar el panel de manera manual
def actualizar_panel():
    st.session_state.data = cargar_datos(st.session_state.temporalidad_seleccionada)
    st.session_state.actualizar_panel = False  # Resetear flag después de la actualización
    
def entrenar_y_predecir(temporalidad):
    data, predictions = f.predict_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
    st.session_state.data = data
    st.session_state.predictions = predictions
    st.session_state.predicciones_entrenadas = True

# Función para actualizar los datos
def actualizar_datos(temporalidad):
    if temporalidad == '1D':
        db.actualizarData1d()
    elif temporalidad == '4H':
        db.actualizarData4h()
    elif temporalidad == '1H':
        db.actualizarData1h()
    elif temporalidad == '5M':
        db.actualizarData5m()

    # Marcar flag para actualizar panel
    #st.session_state.actualizar_panel = True
    #actualizar_panel_directamente()  # Asegurarse de actualizar el panel después de actualizar la data
    
# Función para cargar datos según la temporalidad
def cargar_datos(temporalidad):
    query = f"""
        SELECT date, close, volume, volatility, rsi, ma_5, ma_20, ma_100, CCI, K, D,
               MiddleBand, UpperBand, LowerBand
        FROM btc_{temporalidad.lower()}
    """
    return pd.read_sql_query(query, conn)
# Cargar los datos por defecto si no se han cargado
if st.session_state.data is None:
    st.session_state.data = cargar_datos(st.session_state.temporalidad_seleccionada)

#AGG
# Función para actualizar el panel con nuevos datos
def actualizar_panel_directamente():
    # Recargar los datos y actualizar los valores en session_state
    st.session_state.data = cargar_datos(st.session_state.temporalidad_seleccionada)
    # Resetear flag
    st.session_state.actualizar_panel = False
    
def cortar_data(temporalidad):
    db.cortar_data(temporalidad=temporalidad)
    return None

# Crear gráfico con medias móviles
def agregar_medias_moviles(fig, data):
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_5'].values[-200:], mode='lines', name='MA 5', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_20'].values[-200:], mode='lines', name='MA 20', line=dict(color='purple')))
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_100'].values[-200:], mode='lines', name='MA 100', line=dict(color='red')))
    return fig

# Crear gráfico con bandas de Bollinger
def agregar_bandas_bollinger(fig, data):
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['MiddleBand'].values[-200:], mode='lines', name='Middle Band', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['UpperBand'].values[-200:], mode='lines', name='Upper Band', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['LowerBand'].values[-200:], mode='lines', name='Lower Band', line=dict(color='red')))
    return fig

#AGG
# Función para actualizar el panel de información
def update_info_panel():
    data = st.session_state.data
    st.session_state.actualizar_panel = False
    if data is not None:
        rsi = data['rsi'].iloc[-1]
        cci = data['CCI'].iloc[-1]
        k = data['K'].iloc[-1]
        d = data['D'].iloc[-1]
        styled_table = create_info_table_with_style(rsi=rsi, cci=cci, k=k, d=d)
        st.markdown(styled_table, unsafe_allow_html=True)


# Crear gráficos
def mostrar_graficos(data, predictions, temporalidad):
    fig = go.Figure()
    # Valores reales
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
    # Mostrar predicciones si está seleccionado
    if st.checkbox("Mostrar Predicciones") and predictions is not None:
        largo_predictions = len(predictions) -5 
        # Crear un dataframe de predicciones con fechas alineadas
        if (temporalidad in ['4h','4H']):
            predictions['date'] = pd.date_range(start=data['date'].values[-largo_predictions], periods=len(predictions),freq='4H')
        elif (temporalidad in ['1h','1H']):
            predictions['date'] = pd.date_range(start=data['date'].values[-largo_predictions], periods=len(predictions),freq='1H')
        elif (temporalidad in ['5m','5M']):
            predictions['date'] = pd.date_range(start=data['date'].values[-largo_predictions], periods=len(predictions),freq='5min')
        elif (temporalidad in ['1d','1D']):
            predictions['date'] = pd.date_range(start=data['date'].values[-largo_predictions], periods=len(predictions),freq='1D')
        fig.add_trace(go.Scatter(x=predictions['date'], y=predictions['pred'].values, mode='lines+markers', name='Predicciones', marker=dict(color='green')))
    
    # Mostrar medias móviles si está seleccionado
    if st.checkbox("Mostrar Medias Móviles"):
        fig = agregar_medias_moviles(fig, data)
    
    # Mostrar bandas de Bollinger si está seleccionado
    if st.checkbox("Mostrar Bandas de Bollinger"):
        fig = agregar_bandas_bollinger(fig, data)
    
    # Ajustar diseño y mostrar gráfico
    fig.update_layout(
        title=f'Valores Reales, Predicciones y Indicadores Técnicos - {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white'
    )
    st.plotly_chart(fig)



st.set_page_config(layout="wide")  # Esto asegura que el ancho completo se use
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        #encoded_image = image_file.read()
        encoded_image = base64.b64encode(image_file.read()).decode()  # Codificar la imagen en base64
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{encoded_image});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Llamar a la función para configurar el fondo
set_background('images/fondo_gb.png')  # También puedes usar una ruta local 'assets/background.png'


lista_temporalidades = ['1D', '4H', '1H', '5M']
# Usamos HTML en st.markdown para aplicar color negro al título
st.markdown("<h1 style='color: black;'>Glass Bubble Services</h1>", unsafe_allow_html=True)

# Funciones para estilo de celdas (No son directamente aplicables en Streamlit pero pueden ser usadas para condicionales en visualizaciones)
def estilo_cci(value, lim_inf=-100, lim_sup=100):
    if value < lim_inf:
        return "background-color: green; color: black;"
    elif value > lim_sup:
        return "background-color: red; color: black;"
    else:
        return "background-color: grey; color: black;"

def estilo_porcentuales(value, lim_inf=30, lim_sup=70):
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

# Función para generar la tabla con estilos en HTML
def create_info_table_with_style(rsi, cci,k,d):
    # Crear la tabla en HTML
    html = f"""
    <table style="width:100%; border-collapse: collapse; background-color: #397cc4;">
        <tr>
            <th style="border: 1px solid black; padding: 3px;">Variable</th>
            <th style="border: 1px solid black; padding: 3px;">Valor</th>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Índice de Fuerza Relativa(RSI)</td>
            <td style="border: 1px solid black; padding: 3px; {estilo_porcentuales(rsi)}">{rsi}</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Índice de Canal de Materias Primas(CCI)</td>
            <td style="border: 1px solid black; padding: 3px; {estilo_cci(cci)}">{cci}</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Oscilador Estocástico(OE)(K)</td>
            <td style="border: 1px solid black; padding: 3px; {estilo_porcentuales(k)}">{k}</td>
        </tr>
        <tr>
            <td style="border: 1px solid black; padding: 3px;">Promedio movil del OE(D)</td>
            <td style="border: 1px solid black; padding: 3px; {estilo_porcentuales(d)}">{d}</td>
        </tr>
    </table>
    """
    return html


col1, col2 = st.columns([1, 3])  # Ajusta las proporciones de ancho

with col1:
    # Selección de temporalidad
    temporalidad_seleccionada = st.selectbox(
        "Selecciona la temporalidad",
        ['1D', '4H', '1H', '5M'],
        index=lista_temporalidades.index(st.session_state.temporalidad_seleccionada))
    # Antes de mostrar gráficos o panel de información
    if st.session_state.actualizar_panel:
        #actualizar_panel_directamente()  # Actualiza la información antes de proceder
        update_info_panel()
        st.session_state.actualizar_panel = False  # Restablece el flag a False
    # Guardar la temporalidad seleccionada en el estado de la sesión
    if st.session_state.temporalidad_seleccionada != temporalidad_seleccionada:
        st.session_state.temporalidad_seleccionada = temporalidad_seleccionada
        st.session_state.data = cargar_datos(temporalidad_seleccionada)  # Recargar datos al cambiar temporalidad
    # Llamar a la función de actualización de panel
    #update_info_panel()
    st.subheader("Panel de Información")

    # Mostrar datos técnicos
    if st.session_state.data is not None:
        data = st.session_state.data
        rsi = data['rsi'].iloc[-1]
        cci = data['CCI'].iloc[-1]
        k = data['K'].iloc[-1]
        d = data['D'].iloc[-1]
        styled_table = create_info_table_with_style(rsi=rsi, cci=cci, k=k, d=d)
        st.markdown(styled_table, unsafe_allow_html=True)

    if st.button('Actualizar Datos'):
        actualizar_datos(temporalidad_seleccionada)
        #st.session_state.actualizar_panel = True  # Marcar para actualización del panel
    if st.button('Entrenar Modelo'):
        entrenar_y_predecir(temporalidad_seleccionada)

    if st.button('Recortar datos'):
        cortar_data(temporalidad_seleccionada)
# Mostrar gráficos en la columna derecha
with col2:
    
    #if st.session_state.actualizar_panel:
        #actualizar_panel_directamente()
    if st.session_state.data is not None:
        mostrar_graficos(st.session_state.data, st.session_state.predictions, st.session_state.temporalidad_seleccionada)
    else:
        st.write("Los datos no están disponibles. Por favor, carga los datos o entrena el modelo.")


