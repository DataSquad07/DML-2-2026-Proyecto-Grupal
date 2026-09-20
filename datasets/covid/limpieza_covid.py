# -*- coding: utf-8 -*-
"""
===============================================================================
PRACTICA DEL PRIMER PARCIAL - MACHINE LEARNING
Primer Avance: (a) Limpieza de datos  -  (b) Particion entrenamiento / prueba

Dataset : COVID-19 (covid_19_data.csv)
Area    : Modelos Regresionales
Docente : Lic. Patricia Rodriguez Bilbao
===============================================================================
El script recorre las tres etapas de limpieza vistas en clase:
    1. INCOMPLETITUD  -> valores faltantes
    2. RUIDO          -> valores anomalos, fuera de rango o imposibles
    3. INCONSISTENCIA -> duplicados, formatos y nombres no unificados
Al final realiza la particion 80/20 y guarda el dataset limpio.
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuracion de la salida por consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

RUTA_ENTRADA = 'covid_19_data.csv'
RUTA_SALIDA = 'covid_limpio.csv'
SEMILLA = 42


def separador(titulo):
    """Imprime un encabezado para poder seguir la ejecucion paso a paso."""
    print('\n' + '=' * 78)
    print(titulo)
    print('=' * 78)


# =============================================================================
# PASO 0 - CARGA Y DIAGNOSTICO INICIAL
# =============================================================================
separador('PASO 0 - INFORMACION INICIAL DEL DATASET')

df = pd.read_csv(RUTA_ENTRADA)

print('Dimensiones (filas, columnas):', df.shape)
print('\nEstructura del dataset:')
df.info()

print('\nPrimeros 5 registros:')
print(df.head())

print('\nValores nulos por columna:')
print(df.isnull().sum())

print('\nEstadisticas descriptivas de las variables numericas:')
print(df[['Confirmed', 'Deaths', 'Recovered']].describe())

filas_iniciales = len(df)


# =============================================================================
# PASO 1 - INCOMPLETITUD (tratamiento de valores faltantes)
# =============================================================================
separador('PASO 1 - TRATAMIENTO DE LA INCOMPLETITUD')

print('Nulos detectados antes del tratamiento:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- 1.1 Province/State -------------------------------------------------------
# El nulo NO es un error de carga: corresponde a los paises que reportan la
# informacion de forma nacional, sin desagregar por provincia o estado.
# Eliminar esas filas significaria perder mas del 25% del dataset, por lo que
# se imputa con el nombre del pais, conservando el significado del registro.
nulos_provincia = df['Province/State'].isnull().sum()
df['Province/State'] = df['Province/State'].fillna(df['Country/Region'])
print('\n[1.1] Province/State: {} nulos imputados con el nombre del pais.'
      .format(nulos_provincia))

# --- 1.2 Variables numericas --------------------------------------------------
# Confirmed, Deaths y Recovered son acumulados. Un faltante equivale a que la
# region no reporto ese dia, lo que se interpreta como cero casos registrados.
for col in ['Confirmed', 'Deaths', 'Recovered']:
    faltantes = df[col].isnull().sum()
    if faltantes > 0:
        df[col] = df[col].fillna(0)
        print('[1.2] {}: {} nulos imputados con 0.'.format(col, faltantes))

print('\nNulos restantes luego del paso 1:')
print(df.isnull().sum())


# =============================================================================
# PASO 2 - RUIDO (valores anomalos, imposibles o fuera de rango)
# =============================================================================
separador('PASO 2 - TRATAMIENTO DEL RUIDO')

# --- 2.1 Valores negativos ----------------------------------------------------
# Un conteo de casos nunca puede ser negativo. Cuando aparece se debe a
# correcciones retroactivas de los organismos de salud. Se llevan a cero.
for col in ['Confirmed', 'Deaths', 'Recovered']:
    negativos = (df[col] < 0).sum()
    if negativos > 0:
        df.loc[df[col] < 0, col] = 0
    print('[2.1] {}: {} valores negativos corregidos a 0.'.format(col, negativos))

# --- 2.2 Dependencia de atributos --------------------------------------------
# Regla de negocio: los fallecidos y los recuperados son subconjuntos de los
# casos confirmados, por lo tanto Deaths + Recovered nunca puede superar a
# Confirmed. Los registros que violan la regla son inconsistentes en origen y
# se eliminan por no poder corregirse de forma confiable.
condicion_imposible = (df['Deaths'] + df['Recovered']) > df['Confirmed']
registros_imposibles = condicion_imposible.sum()
df = df[~condicion_imposible].copy()
print('[2.2] Registros eliminados por violar Deaths + Recovered <= Confirmed: {}'
      .format(registros_imposibles))

# --- 2.3 Deteccion de valores atipicos (metodo del rango intercuartilico) ----
# Los outliers se REPORTAN pero NO se eliminan: en un fenomeno de propagacion
# los picos son informacion real y no error de medicion. La deteccion sirve
# para documentar la distribucion antes de la etapa de modelado.
print('\n[2.3] Deteccion de valores atipicos (solo informativa):')
for col in ['Confirmed', 'Deaths', 'Recovered']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    riq = q3 - q1
    limite_inferior = q1 - 1.5 * riq
    limite_superior = q3 + 1.5 * riq
    atipicos = ((df[col] < limite_inferior) | (df[col] > limite_superior)).sum()
    print('      {:<10} Q1={:>12.1f}  Q3={:>12.1f}  atipicos={}'
          .format(col, q1, q3, atipicos))

# --- 2.4 Conversion de tipos --------------------------------------------------
# Los conteos de personas deben ser enteros, no flotantes.
df[['Confirmed', 'Deaths', 'Recovered']] = \
    df[['Confirmed', 'Deaths', 'Recovered']].astype('int64')
print('\n[2.4] Confirmed, Deaths y Recovered convertidos a entero.')


# =============================================================================
# PASO 3 - INCONSISTENCIA (duplicados, formatos y nomenclatura)
# =============================================================================
separador('PASO 3 - TRATAMIENTO DE LA INCONSISTENCIA')

# --- 3.1 Normalizacion de texto ----------------------------------------------
# Se eliminan espacios sobrantes al inicio y al final. En el dataset original
# existen entradas como " Azerbaijan" que el modelo tomaria como un pais
# distinto de "Azerbaijan".
for col in ['Province/State', 'Country/Region']:
    df[col] = df[col].astype(str).str.strip()
print('[3.1] Espacios sobrantes eliminados en las columnas de texto.')

# --- 3.2 Unificacion de nombres de paises ------------------------------------
# El mismo pais aparece escrito de varias formas segun la fuente que reporto.
equivalencias_pais = {
    'Mainland China': 'China',
    'Hong Kong SAR': 'Hong Kong',
    'Macao SAR': 'Macau',
    'Macao': 'Macau',
    'UK': 'United Kingdom',
    'US': 'United States',
    'Republic of Ireland': 'Ireland',
    'Iran (Islamic Republic of)': 'Iran',
    'Republic of Korea': 'South Korea',
    'Korea, South': 'South Korea',
    'Taiwan*': 'Taiwan',
    'Viet Nam': 'Vietnam',
    'Russian Federation': 'Russia',
    'Bahamas, The': 'Bahamas',
    'The Bahamas': 'Bahamas',
    'Gambia, The': 'Gambia',
    'The Gambia': 'Gambia',
    'Czech Republic': 'Czechia',
    'Cape Verde': 'Cabo Verde',
    'Ivory Coast': "Cote d'Ivoire",
}
paises_antes = df['Country/Region'].nunique()
df['Country/Region'] = df['Country/Region'].replace(equivalencias_pais)
paises_despues = df['Country/Region'].nunique()
print('[3.2] Nomenclatura de paises unificada: {} -> {} valores unicos.'
      .format(paises_antes, paises_despues))

# --- 3.3 Formato de fechas ----------------------------------------------------
# Las fechas venian como texto y con formatos mezclados. Se convierten a un
# unico tipo datetime para poder derivar variables temporales.
df['ObservationDate'] = pd.to_datetime(df['ObservationDate'],
                                       format='mixed', errors='coerce')
df['Last Update'] = pd.to_datetime(df['Last Update'],
                                   format='mixed', errors='coerce')
fechas_invalidas = df['ObservationDate'].isnull().sum()
if fechas_invalidas > 0:
    df = df.dropna(subset=['ObservationDate'])
print('[3.3] Fechas convertidas a datetime. Registros con fecha ilegible '
      'eliminados: {}'.format(fechas_invalidas))

# --- 3.4 Registros duplicados -------------------------------------------------
# La llave logica del dataset es la combinacion fecha + pais + provincia.
# Si se repite, se trata de un reporte cargado dos veces; se conserva el
# ultimo por ser el mas actualizado.
duplicados = df.duplicated(
    subset=['ObservationDate', 'Country/Region', 'Province/State']).sum()
df = df.drop_duplicates(
    subset=['ObservationDate', 'Country/Region', 'Province/State'],
    keep='last').copy()
print('[3.4] Registros duplicados eliminados: {}'.format(duplicados))

# --- 3.5 Columna sin aporte al modelo ----------------------------------------
# SNo es solo un correlativo de carga; no aporta informacion predictiva y
# ademas quedo desordenado luego de las eliminaciones.
df = df.drop(columns=['SNo'])
print('[3.5] Columna SNo eliminada por ser un identificador correlativo.')

# --- 3.6 Reindexado -----------------------------------------------------------
df = df.reset_index(drop=True)


# =============================================================================
# PASO 4 - VERIFICACION DE LA LIMPIEZA
# =============================================================================
separador('PASO 4 - DATOS LIMPIOS CORRECTAMENTE')

print('Valores nulos restantes:')
print(df.isnull().sum())

print('\nRegistros duplicados restantes:',
      df.duplicated(subset=['ObservationDate', 'Country/Region',
                            'Province/State']).sum())
print('Valores negativos restantes:',
      int((df[['Confirmed', 'Deaths', 'Recovered']] < 0).sum().sum()))
print('Registros que violan la regla de negocio:',
      int(((df['Deaths'] + df['Recovered']) > df['Confirmed']).sum()))

print('\nFilas iniciales : {}'.format(filas_iniciales))
print('Filas finales   : {}'.format(len(df)))
print('Filas removidas : {} ({:.2f}% del total)'
      .format(filas_iniciales - len(df),
              (filas_iniciales - len(df)) / filas_iniciales * 100))

df.to_csv(RUTA_SALIDA, index=False)
print('\nDataset limpio guardado en:', RUTA_SALIDA)


# =============================================================================
# PASO 5 - PARTICION EN ENTRENAMIENTO Y PRUEBA (80 / 20)
# =============================================================================
separador('PASO 5 - RESULTADO DE LA PARTICION')

# Variable objetivo (etiqueta): Deaths
# Variables predictoras (caracteristicas): Confirmed y Recovered
X = df[['Confirmed', 'Recovered']]
y = df['Deaths']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEMILLA,   # semilla fija -> particion reproducible
    shuffle=True
)

print('Total de filas procesadas          :', len(df))
print('Muestras de Entrenamiento (80%)    :', len(X_train))
print('Muestras de Prueba (20%)           :', len(X_test))
print('Caracteristicas utilizadas         :', list(X.columns))
print('Etiqueta a predecir                :', y.name)
