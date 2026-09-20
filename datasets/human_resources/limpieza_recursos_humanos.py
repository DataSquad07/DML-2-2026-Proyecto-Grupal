# -*- coding: utf-8 -*-
"""
===============================================================================
PRACTICA DEL PRIMER PARCIAL - MACHINE LEARNING
Primer Avance: (a) Limpieza de datos  -  (b) Particion entrenamiento / prueba

Dataset : Human Resources (HRDataset_v14.csv)
Area    : Modelos de Arboles de Decision
Docente : Lic. Patricia Rodriguez Bilbao
===============================================================================
El script recorre las tres etapas de limpieza vistas en clase:
    1. INCOMPLETITUD  -> valores faltantes
    2. RUIDO          -> valores anomalos, fuera de rango o imposibles
    3. INCONSISTENCIA -> duplicados, formatos y categorias no unificadas
Al final realiza la particion 80/20 estratificada y guarda el dataset limpio.
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)

RUTA_ENTRADA = 'HRDataset_v14.csv'
RUTA_SALIDA = 'recursos_humanos_limpio.csv'
SEMILLA = 42


def separador(titulo):
    print('\n' + '=' * 78)
    print(titulo)
    print('=' * 78)


# =============================================================================
# PASO 0 - CARGA Y DIAGNOSTICO INICIAL
# =============================================================================
separador('PASO 0 - INFORMACION INICIAL (HUMAN RESOURCES)')

df = pd.read_csv(RUTA_ENTRADA)

print('Dimensiones (filas, columnas):', df.shape)
print('\nEstructura del dataset:')
df.info()

print('\nValores nulos iniciales:')
print(df.isnull().sum()[df.isnull().sum() > 0])

print('\nEstadisticas de las variables numericas relevantes:')
print(df[['Salary', 'EngagementSurvey', 'EmpSatisfaction',
          'Absences', 'DaysLateLast30']].describe())

filas_iniciales = len(df)


# =============================================================================
# PASO 1 - INCOMPLETITUD (tratamiento de valores faltantes)
# =============================================================================
separador('PASO 1 - TRATAMIENTO DE LA INCOMPLETITUD')

print('Nulos detectados antes del tratamiento:')
print(df.isnull().sum()[df.isnull().sum() > 0])

# --- 1.1 DateofTermination ----------------------------------------------------
# Los 207 nulos NO son un error. Corresponden a los empleados que siguen
# trabajando en la empresa y por eso no tienen fecha de baja registrada.
# Imputarlos con una fecha inventada corromperia el dato, por lo que se crea
# una variable booleana que expresa el significado real del faltante y luego
# se rellena la fecha con un marcador neutro.
nulos_baja = df['DateofTermination'].isnull().sum()
df['EmpleadoActivo'] = df['DateofTermination'].isnull().astype(int)
df['DateofTermination'] = df['DateofTermination'].fillna('Activo')
print('\n[1.1] DateofTermination: {} nulos interpretados como empleado activo.'
      .format(nulos_baja))
print('      Se genero la variable EmpleadoActivo (1 = activo, 0 = dado de baja).')

# --- 1.2 ManagerID ------------------------------------------------------------
# Los 8 nulos si son un error de carga: el empleado tiene ManagerName pero no
# el codigo numerico. Se recupera el ID a partir del nombre del jefe usando el
# resto de registros del mismo supervisor.
nulos_manager = df['ManagerID'].isnull().sum()
mapa_manager = (df.dropna(subset=['ManagerID'])
                  .drop_duplicates('ManagerName')
                  .set_index('ManagerName')['ManagerID'])
df['ManagerID'] = df['ManagerID'].fillna(df['ManagerName'].map(mapa_manager))

# Si aun quedara alguno sin recuperar, se completa con la moda de la columna.
if df['ManagerID'].isnull().sum() > 0:
    df['ManagerID'] = df['ManagerID'].fillna(df['ManagerID'].mode()[0])
df['ManagerID'] = df['ManagerID'].astype(int)
print('[1.2] ManagerID: {} nulos recuperados a partir de ManagerName.'
      .format(nulos_manager))

print('\nNulos restantes luego del paso 1:')
print(df.isnull().sum().sum(), 'valores nulos en todo el dataset.')


# =============================================================================
# PASO 2 - RUIDO (valores anomalos, imposibles o fuera de rango)
# =============================================================================
separador('PASO 2 - TRATAMIENTO DEL RUIDO')

# --- 2.1 Fechas de nacimiento imposibles -------------------------------------
# El archivo guarda el anio con dos digitos (mm/dd/yy). Al convertirlo, pandas
# interpreta "62" como 2062, lo que genera empleados con edad negativa.
df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%y', errors='coerce')
futuras = (df['DOB'] > pd.Timestamp('today')).sum()
df.loc[df['DOB'] > pd.Timestamp('today'), 'DOB'] -= pd.DateOffset(years=100)
print('[2.1] Fechas de nacimiento en el futuro corregidas: {}'.format(futuras))

# Se calcula la edad para poder validar el rango laboral.
df['Edad'] = ((pd.Timestamp('today') - df['DOB']).dt.days / 365.25).astype(int)
fuera_rango = ((df['Edad'] < 18) | (df['Edad'] > 75)).sum()
print('      Edades fuera del rango laboral valido (18-75 anios): {}'
      .format(fuera_rango))

# --- 2.2 Salarios ilogicos ----------------------------------------------------
salarios_invalidos = (df['Salary'] <= 0).sum()
if salarios_invalidos > 0:
    df = df[df['Salary'] > 0].copy()
print('[2.2] Registros eliminados por salario menor o igual a cero: {}'
      .format(salarios_invalidos))

# --- 2.3 Rangos de las escalas de evaluacion ---------------------------------
# EmpSatisfaction esta definida de 1 a 5 y EngagementSurvey de 1 a 5.
# Cualquier valor fuera de ese rango es ruido de captura.
rangos = {'EmpSatisfaction': (1, 5), 'EngagementSurvey': (1, 5)}
for col, (minimo, maximo) in rangos.items():
    invalidos = ((df[col] < minimo) | (df[col] > maximo)).sum()
    df[col] = df[col].clip(lower=minimo, upper=maximo)
    print('[2.3] {}: {} valores fuera del rango [{}, {}] ajustados al limite.'
          .format(col, invalidos, minimo, maximo))

# --- 2.4 Deteccion de atipicos en el salario (rango intercuartilico) ---------
# Los salarios altos corresponden a cargos gerenciales reales, por lo que se
# reportan pero no se eliminan.
q1 = df['Salary'].quantile(0.25)
q3 = df['Salary'].quantile(0.75)
riq = q3 - q1
limite_superior = q3 + 1.5 * riq
atipicos_salario = (df['Salary'] > limite_superior).sum()
print('[2.4] Salarios atipicos detectados (> {:.0f}): {} '
      '(se conservan, corresponden a cargos directivos).'
      .format(limite_superior, atipicos_salario))


# =============================================================================
# PASO 3 - INCONSISTENCIA (duplicados, formatos y categorias)
# =============================================================================
separador('PASO 3 - TRATAMIENTO DE LA INCONSISTENCIA')

# --- 3.1 Normalizacion de texto ----------------------------------------------
# Varios nombres traen espacios al final y el separador de apellido no es
# uniforme. Se limpian todas las columnas de tipo texto.
columnas_texto = df.select_dtypes(include='object').columns
for col in columnas_texto:
    df[col] = df[col].astype(str).str.strip()
print('[3.1] Espacios sobrantes eliminados en {} columnas de texto.'
      .format(len(columnas_texto)))

# --- 3.2 Categorias escritas de distinta forma -------------------------------
# La columna HispanicLatino mezcla mayusculas y minusculas ("Yes"/"yes",
# "No"/"no"), lo que el modelo interpretaria como cuatro categorias distintas.
antes_hl = df['HispanicLatino'].nunique()
df['HispanicLatino'] = df['HispanicLatino'].str.capitalize()
print('[3.2] HispanicLatino: {} categorias unificadas a {}.'
      .format(antes_hl, df['HispanicLatino'].nunique()))
print('      Valores finales:', sorted(df['HispanicLatino'].unique()))

# El nombre del departamento tambien traia espacios internos irregulares.
df['Department'] = df['Department'].str.replace(r'\s+', ' ', regex=True)
print('      Department normalizado:', df['Department'].nunique(),
      'departamentos unicos.')

# --- 3.3 Dependencia de atributos --------------------------------------------
# Termd indica si el empleado fue dado de baja. Debe ser coherente con la
# fecha de baja: si Termd = 1 tiene que existir DateofTermination.
incoherentes = ((df['Termd'] == 1) & (df['DateofTermination'] == 'Activo')).sum()
df.loc[(df['Termd'] == 1) & (df['DateofTermination'] == 'Activo'),
       'EmpleadoActivo'] = 0
print('[3.3] Registros con Termd = 1 pero sin fecha de baja: {} '
      '(coherencia corregida).'.format(incoherentes))

# --- 3.4 Registros duplicados -------------------------------------------------
# EmpID es la llave primaria: no puede repetirse.
duplicados_id = df.duplicated(subset=['EmpID']).sum()
df = df.drop_duplicates(subset=['EmpID'], keep='first').copy()
duplicados_totales = df.duplicated().sum()
df = df.drop_duplicates().copy()
print('[3.4] Duplicados por EmpID eliminados: {}'.format(duplicados_id))
print('      Filas completamente duplicadas eliminadas: {}'
      .format(duplicados_totales))

# --- 3.5 Formato unico de fechas ---------------------------------------------
df['DateofHire'] = pd.to_datetime(df['DateofHire'], errors='coerce')
df['LastPerformanceReview_Date'] = pd.to_datetime(
    df['LastPerformanceReview_Date'], errors='coerce')
print('[3.5] Columnas de fecha convertidas a un unico formato datetime.')

# --- 3.6 Columnas sin aporte predictivo --------------------------------------
# Se descartan los identificadores y los campos redundantes (las columnas
# terminadas en ID duplican la informacion de su columna descriptiva).
columnas_a_eliminar = ['Employee_Name', 'Zip', 'MarriedID', 'MaritalStatusID',
                       'DeptID', 'PerfScoreID', 'EmpStatusID', 'PositionID',
                       'FromDiversityJobFairID']
columnas_a_eliminar = [c for c in columnas_a_eliminar if c in df.columns]
df = df.drop(columns=columnas_a_eliminar)
print('[3.6] Columnas eliminadas por ser identificadores o redundantes: {}'
      .format(len(columnas_a_eliminar)))

df = df.reset_index(drop=True)


# =============================================================================
# PASO 4 - VERIFICACION DE LA LIMPIEZA
# =============================================================================
separador('PASO 4 - DATOS LIMPIOS CORRECTAMENTE')

print('Valores nulos restantes por columna (solo se listan los mayores a 0):')
nulos_finales = df.isnull().sum()
print(nulos_finales[nulos_finales > 0] if nulos_finales.sum() > 0
      else 'Ninguno. El dataset no presenta valores nulos.')

print('\nRegistros duplicados restantes :', df.duplicated().sum())
print('Filas iniciales                :', filas_iniciales)
print('Filas finales                  :', len(df))
print('Columnas finales               :', df.shape[1])

df.to_csv(RUTA_SALIDA, index=False)
print('\nDataset limpio guardado en:', RUTA_SALIDA)


# =============================================================================
# PASO 5 - PARTICION EN ENTRENAMIENTO Y PRUEBA (80 / 20)
# =============================================================================
separador('PASO 5 - RESULTADO DE LA PARTICION')

# Variable objetivo (etiqueta): Termd (1 = el empleado se fue, 0 = sigue)
# Se seleccionan caracteristicas numericas para el arbol de decision.
caracteristicas = ['Salary', 'EngagementSurvey', 'EmpSatisfaction',
                   'SpecialProjectsCount', 'DaysLateLast30', 'Absences',
                   'Edad', 'ManagerID']

X = df[caracteristicas]
y = df['Termd']

# stratify garantiza que la proporcion de bajas se mantenga en ambos conjuntos.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEMILLA,
    stratify=y
)

print('Total de registros              :', len(df))
print('Muestras de Entrenamiento (80%) :', len(X_train))
print('Muestras de Prueba (20%)        :', len(X_test))
print('Caracteristicas utilizadas      :', len(caracteristicas))
print('Etiqueta a predecir             : Termd')
print('\nDistribucion de la etiqueta en entrenamiento:')
print(y_train.value_counts(normalize=True).round(3))
print('\nDistribucion de la etiqueta en prueba:')
print(y_test.value_counts(normalize=True).round(3))
