import pandas as pd
import sqlite3 as sql
import matplotlib.pyplot as plt
import plotly_express as px
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import seaborn as sns

from lightgbm import LGBMRegressor
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.preprocessing import StandardScaler
import joblib # pip install joblib
import pandas as pd
#from scripts.entrenar_modelo import entrenar_modelo
#from scripts.guardar_modelo import guardar_modelo
#from utils.cargar_datos import cargar_datos

conn = sql.connect('btc.db')
cursor = conn.cursor()

def reentrenar_modelo(db_path, modelo_path):
    df = cargar_datos(db_path)
    modelo = entrenar_modelo(df)
    guardar_modelo(modelo, modelo_path)
    return None
