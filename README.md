# Glass Bubble

<img src="images/bola_cristal.jfif" width="300" height="300">

### Glass Bubble se destaca en el mercado por ofrecer servicios de monitorio y seguimiento de modelos de ML y DP.

## Contexto
#### Bull Market Broker(BMB) llego a la conclusion de que la adaptacion de su plataforma para incluir criptomonedas era algo por concluir.

Bull Market Broker busca expandir su base de usuarios y capitalizar el creciente interés en las criptomonedas. Al incorporar criptomonedas a su plataforma, BMB apunta a atraer a un público más joven y tecnológicamente avanzado, que busca invertir en activos digitales con alto potencial de crecimiento.

El público objetivo de BMB son principalmente traders activos y entusiastas de las criptomonedas con conocimientos básicos de análisis técnico. Estos usuarios buscan herramientas que les permitan realizar un seguimiento en tiempo real de los precios, identificar patrones y tomar decisiones de inversión informadas.

#### Bitcoin ha sido seleccionado como el primer activo debido a su alta liquidez, reconocimiento de marca y madurez en comparación con otras criptomonedas. Además, Bitcoin es ampliamente considerado como el "oro digital" y sirve como una referencia para el resto del mercado de criptomonedas.

#### El objetivo del cliente es un tablero interactivo con un panel donde se encuentren las variables que considere mas importantes a la hora de evaluar la serie.
#### Por lo que, el alcance del proyecto se extendera solamente hasta bitcoin con 4 temporalidades:

- 1 dia 
- 4 horas
- 1 hora
- 5 min.

Se consideraran otras mejoras del tablero en la seccion de Upgrades.

## Objetivos

Tablero interactivo con informacion acerca del activo.

Objetivos específicos: ¿Qué acciones espera que los usuarios realicen con el tablero? ¿Tomar decisiones de inversión, monitorear el mercado, comparar activos? Definir objetivos claros ayudará a orientar el diseño y las funcionalidades.

Métricas de éxito: ¿Cómo se medirá el éxito del tablero? ¿Número de usuarios, tiempo de uso, precisión de las predicciones? Establecer métricas te permitirá evaluar el desempeño del proyecto y realizar ajustes si es necesario.


## Entregables

**Tablero interactivo**

El tablero se realizara usando Dash y demas librerias de python.

**Componentes**:
- 4 Graficas(24h,4h,1h,5m) interactivas con valor real y prediccion futura.
- Botonera:

- a) boton actualizar: Al ejecutarse, se realizara una llamada a la api para la extraccion
de los datos faltantes, se transformaran y cargaran en una base de datos local de sqlite3.
- b) boton reentrenar: Al ejecutarse, se entrenara un modelo de machine learning sobre
los ultimos datos del dataframe y se actualizaran las predicciones sobre los datos
en tiempo real.
- c) boton cortar: Al ejecutarse, se reduciran los datos cargados en la base sql, con el fin
de no almacenar datos mas antiguos de lo que el modelo requiere.

- Panel: panel con valores de las variables que describen diferentes caracteristicas del activo en ese momento especifico, pudiendo ser alternado entre los 4 marcos temporales que tienen los graficos.

### Interaccion del usuario

El usuario atravez del tablero puede ejecutar 4 acciones:

- Actualizar la base de datos, al ejecutarse, se realizara una llamada a la api para la extraccion de los datos faltantes, se transformaran y almacenaran en la base de datos.

- Reentrenar modelo, al ejecutarse se volvera a entrenar el modelo con los ultimos datos y generar predicciones en tiempo real.

- Depurar, al ejecutarse, se reduciran los datos cargados en la base de datos con el fin de no almacenar datos mas antiguos de lo que el modelo requiere.

- Eleccion del marco temporal del tablero.

El tablero incluye un panel interactivo que permite a los usuarios explorar los datos del activo en diferentes escalas temporales. Al seleccionar un marco temporal determinado (1 día, 4 horas, 1 hora o 5 minutos), el panel se actualizará automáticamente para mostrar los indicadores más importantes calculados sobre los datos de ese período.

### Logica de las predicciones

#### Aclaracion: El precio del bitcoin es influenciado por muchos valores externos, y es verdaderamente complicado tambien asi, demostrar que esos valores externos son realmente la causa del valor del activo y no una casualidad debido a que: *correlacion no implica causalidad*. Dicho esto, el modelo utilizado fue un ForecasterAutoregresivo.

El precio del Bitcoin es altamente volátil debido a:
- Su naturaleza descentralizada: Sin un banco central que controle la oferta, el mercado es más susceptible a cambios bruscos.
- Factores externos impredecibles: Noticias, regulaciones y la psicología de los inversores influyen directamente en el precio.
- Al ser un activo nuevo, los patrones históricos no siempre son confiables para predecir su comportamiento.

Explica la lógica de las predicciones: ¿Qué tipo de modelo de machine learning se utilizará? ¿Cuáles son las características que se utilizarán para realizar las predicciones?
Detalla la arquitectura: ¿Cómo se conectan los diferentes componentes del tablero? ¿Cuál es el flujo de datos desde la API hasta la interfaz de usuario?

#### ¿Porque se eligio un ForecasterAutoregresivo?

- Captura de dependencias temporales: Los modelos autoregresivos están diseñados para capturar la dependencia entre los valores pasados y futuros de una serie de tiempo, lo que es fundamental para realizar predicciones precisas.
- Manejo de series de tiempo con estacionalidad: LightGBM puede manejar fácilmente la estacionalidad en las series de tiempo, lo que es común en muchos conjuntos de datos reales.
- Alta precisión: LightGBM es un algoritmo de boosting muy eficiente y preciso, lo que lo convierte en una excelente opción para modelos de predicción de series de tiempo.
- Flexibilidad: Permite la creación de modelos complejos que combinan múltiples características y técnicas de modelado.

## Upgrades:

- Posible mejora, al integrar todos los pasos las extracciones y la generacion de la base de datos, en un mismo archivo.py.

- Emplear una mejora respecto a la extraccion de datos.
Las funciones de actualizacion de los datos, presentan una posible mejora respecto a que al terminar de leer los datos, se vuelve a leer la totalidad del dataset y se aplica nuevamente el proceso de creacion de columnas calculadas. Considerar la forma de que esto solo se realice sobre los ultimos datos solamente y concatenando ambos dataframes(antiguo y nuevo).

- Considerar funciones que puedan ser ejecutadas por fuera del main, para alivinar el proceso de ejecucion de las funciones en el archivo main.py

- Emplear una eficientizacion respecto a las variables que se manipulan dentro de main para eliminar variables o archivos que puedan acumular datos en memoria.


