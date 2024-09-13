import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3 as sql
import Scripts.db_crud as db
import Scripts.model_script as f

import plotly.graph_objects as go
#streamlit run main.py
# Conectar a la base de datos
conn = sql.connect('Data/db/btc.db')
cursor = conn.cursor()

# Función para cargar datos según la temporalidad
def cargar_datos(temporalidad):
    query = f"""
        SELECT date, close, volume, volatility, rsi, ma_5, ma_20, ma_100, CCI, K, D,
               MiddleBand, UpperBand, LowerBand
        FROM btc_{temporalidad.lower()}
    """
    return pd.read_sql_query(query, conn)


def entrenar_y_predecir(temporalidad):
    #data = func.load_and_prepare_data_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
    data, predictions = f.predict_sin_exog('Data/db/btc.db', temporalidad=temporalidad)
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
        template='plotly_white',
        plot_bgcolor='skyblue',  # Fondo del gráfico (área donde están las líneas)
        paper_bgcolor='gray'
    )
    return fig

# Crear gráfico de predicciones
def crear_grafico_predicciones(data, predictions, temporalidad):
    fig = go.Figure()
    predicciones = predictions
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], line=dict(color='gray'),
                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
    fig.add_trace(go.Scatter(x=predicciones.index, y=predicciones.pred.values, line=dict(color='gray'),
                             mode='lines+markers', name='Predicciones en 5 Pasos', marker=dict(color='green')))
    
    fig.update_layout(
        title=f'Predicciones {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white',
        plot_bgcolor='skyblue',  # Fondo del gráfico (área donde están las líneas)
        paper_bgcolor='gray'
    )
    return fig


# Función para crear gráfico con predicciones
def crear_grafico_valores_predichos(data, predictions, temporalidad):
    fig = go.Figure()
    # Valores reales
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], 
                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))
    # Predicciones
    fig.add_trace(go.Scatter(x=predictions.index, y=predictions['pred'].values, 
                             mode='lines+markers', name='Predicciones', marker=dict(color='green')))
    
    fig.update_layout(
        title=f'Valores Reales y Predicciones {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white',
        plot_bgcolor='skyblue',
        paper_bgcolor='gray'
    )
    return fig

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

# Crear gráfico con todas las capas: valores reales, predicciones y medias móviles
def mostrar_graficos(data, predictions, temporalidad):
    # Crear gráfico base con valores reales
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['close'].values[-200:], 
                             mode='lines+markers', name='Valores Reales', marker=dict(color='yellow', symbol='cross')))

    # Checkbox para mostrar predicciones
    mostrar_predicciones = st.checkbox("Mostrar Predicciones")
    if mostrar_predicciones and predictions is not None:
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['pred'].values, 
                                 mode='lines+markers', name='Predicciones', marker=dict(color='green')))
    
    # Checkbox para mostrar medias móviles
    mostrar_ma = st.checkbox("Mostrar Medias Móviles")
    if mostrar_ma:
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_5'].values[-200:], mode='lines', name='MA 5', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_20'].values[-200:], mode='lines', name='MA 20', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['ma_100'].values[-200:], mode='lines', name='MA 100', line=dict(color='red')))

    # Checkbox para mostrar bandas de Bollinger
    mostrar_bandas = st.checkbox("Mostrar Bandas de Bollinger")
    if mostrar_bandas:
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['MiddleBand'].values[-200:], mode='lines', name='Middle Band', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['UpperBand'].values[-200:], mode='lines', name='Upper Band', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data['date'][-200:], y=data['LowerBand'].values[-200:], mode='lines', name='Lower Band', line=dict(color='red')))

    # Ajustar diseño
    fig.update_layout(
        title=f'Valores Reales, Predicciones y Medias Móviles {temporalidad}',
        xaxis_title='Fecha',
        yaxis_title='Precio',
        hovermode='x',
        template='plotly_white',
        plot_bgcolor='skyblue',
        paper_bgcolor='gray'
    )
    
    # Mostrar gráfico
    st.plotly_chart(fig)


import base64  # Para codificar la imagen a base64
st.set_page_config(layout="wide")  # Esto asegura que el ancho completo se use
# Definir la función para agregar el fondo
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



#st.title("Glass Bubble Services")
#st.title("<font color='#000000'> :glass_of_wine: Glass Bubble Services </font>")
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

# Crear tablas de información para mostrar
def create_info_table(volume, volatility):
    return pd.DataFrame({
        "Variable": ["Volumen", "Volatilidad"],
        "Valor": [volume, volatility]
    })


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


# Crear columnas: la primera columna será el panel de información y la segunda columna los gráficos
col1, col2 = st.columns([1, 3])  # Ajusta las proporciones de ancho

# Mostrar el panel de información en la columna izquierda
with col1:
    #temporalidad_seleccionada = st.selectbox("Selecciona la temporalidad", lista_temporalidades)
    #index = 0 carga el primer valor de la lista, es decir, por defecto el valor de "1d"
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
        
        rsi = data['rsi'].iloc[-1]
        cci = data['CCI'].iloc[-1]
        k = data['K'].iloc[-1]
        d = data['D'].iloc[-1]
        # Generar la tabla con estilo y mostrarla con HTML
        styled_table = create_info_table_with_style(rsi=rsi, cci=cci,k=k,d=d)
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

with col2:
    if data is not None:
        data, predictions = entrenar_y_predecir(temporalidad_seleccionada)
        mostrar_graficos(data, predictions, temporalidad_seleccionada)



#with col2:
#    st.subheader(f"Gráfico - {temporalidad_seleccionada}")
#    
#    # Recuperar los datos y predicciones de session_state
#    data = st.session_state.get('data', data)
#    predictions = st.session_state.get('predictions')
#    
#    # Llamar a la función para mostrar los gráficos con las opciones de checkboxes
#    mostrar_graficos(data, temporalidad_seleccionada)
#    
#    # Mostrar predicciones si están disponibles
#    if predictions is not None:
#        st.subheader("Predicciones")
#        fig_pred = crear_grafico_predicciones(data=data,predictions=predictions, temporalidad=temporalidad_seleccionada)
#        st.plotly_chart(fig_pred)