# -*- coding: utf-8 -*-
"""
===============================================================================
PRACTICA DE MACHINE LEARNING - RECURSOS HUMANOS
Procesamiento Completo: (a) Limpieza de datos  -  (b) Partición 80/20

Dataset : Human Resources (HRDataset_v14.csv)
Área    : Modelos de Árboles de Decisión / Clasificación
Docente : Lic. Patricia Rodríguez Bilbao
===============================================================================
El script recorre las tres etapas de limpieza:
    1. INCOMPLETITUD  -> Valores faltantes
    2. RUIDO          -> Valores anómalos, fuera de rango o imposibles
    3. INCONSISTENCIA -> Duplicados, formatos y categorías no unificadas
Al final realiza la partición 80/20 estratificada y guarda el dataset limpio.
===============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuración de la salida por consola
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)


def separador(titulo):
    """Imprime un encabezado para poder seguir la ejecución paso a paso."""
    print('\n' + '=' * 78)
    print(titulo)
    print('=' * 78)


def limpiar_y_preparar_hr(ruta_entrada='datasets/human_resources/HRDataset_v14.csv',
                          ruta_salida='datasets/human_resources/recursos_humanos_limpio.csv',
                          semilla=42):
    """
    Realiza todo el flujo de limpieza y partición para el dataset de Recursos Humanos.
    Devuelve: X_train, X_test, y_train, y_test
    """

    # =========================================================================
    # PASO 0 - CARGA Y DIAGNÓSTICO INICIAL
    # =========================================================================
    separador('PASO 0 - INFORMACIÓN INICIAL (HUMAN RESOURCES)')

    df = pd.read_csv(ruta_entrada)

    print('Dimensiones (filas, columnas):', df.shape)
    print('\nEstructura del dataset:')
    df.info()

    print('\nValores nulos iniciales:')
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print('\nEstadísticas de las variables numéricas relevantes:')
    print(df[['Salary', 'EngagementSurvey', 'EmpSatisfaction',
              'Absences', 'DaysLateLast30']].describe())

    filas_iniciales = len(df)

    # =========================================================================
    # PASO 1 - INCOMPLETITUD (Tratamiento de valores faltantes)
    # =========================================================================
    separador('PASO 1 - TRATAMIENTO DE LA INCOMPLETITUD')

    print('Nulos detectados antes del tratamiento:')
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # --- 1.1 DateofTermination -----------------------------------------------
    nulos_baja = df['DateofTermination'].isnull().sum()
    df['EmpleadoActivo'] = df['DateofTermination'].isnull().astype(int)
    df['DateofTermination'] = df['DateofTermination'].fillna('Activo')
    print('\n[1.1] DateofTermination: {} nulos interpretados como empleado activo.'
          .format(nulos_baja))
    print('      Se generó la variable EmpleadoActivo (1 = activo, 0 = dado de baja).')

    # --- 1.2 ManagerID -------------------------------------------------------
    nulos_manager = df['ManagerID'].isnull().sum()
    mapa_manager = (df.dropna(subset=['ManagerID'])
                      .drop_duplicates('ManagerName')
                      .set_index('ManagerName')['ManagerID'])
    df['ManagerID'] = df['ManagerID'].fillna(df['ManagerName'].map(mapa_manager))

    # Si aún quedara alguno sin recuperar, se completa con la moda.
    if df['ManagerID'].isnull().sum() > 0:
        df['ManagerID'] = df['ManagerID'].fillna(df['ManagerID'].mode()[0])
    df['ManagerID'] = df['ManagerID'].astype(int)
    print('[1.2] ManagerID: {} nulos recuperados a partir de ManagerName.'
          .format(nulos_manager))

    print('\nNulos restantes luego del paso 1:')
    print(df.isnull().sum().sum(), 'valores nulos en todo el dataset.')

    # =========================================================================
    # PASO 2 - RUIDO (Valores anómalos, imposibles o fuera de rango)
    # =========================================================================
    separador('PASO 2 - TRATAMIENTO DEL RUIDO')

    # --- 2.1 Fechas de nacimiento imposibles ---------------------------------
    df['DOB'] = pd.to_datetime(df['DOB'], format='%m/%d/%y', errors='coerce')
    futuras = (df['DOB'] > pd.Timestamp('today')).sum()
    df.loc[df['DOB'] > pd.Timestamp('today'), 'DOB'] -= pd.DateOffset(years=100)
    print('[2.1] Fechas de nacimiento en el futuro corregidas: {}'.format(futuras))

    # Cálculo de edad para validación laboral
    df['Edad'] = ((pd.Timestamp('today') - df['DOB']).dt.days / 365.25).astype(int)
    fuera_rango = ((df['Edad'] < 18) | (df['Edad'] > 75)).sum()
    print('      Edades fuera del rango laboral válido (18-75 años): {}'
          .format(fuera_rango))

    # --- 2.2 Salarios ilógicos -----------------------------------------------
    salarios_invalidos = (df['Salary'] <= 0).sum()
    if salarios_invalidos > 0:
        df = df[df['Salary'] > 0].copy()
    print('[2.2] Registros eliminados por salario menor o igual a cero: {}'
          .format(salarios_invalidos))

    # --- 2.3 Rangos de las escalas de evaluación ----------------------------
    rangos = {'EmpSatisfaction': (1, 5), 'EngagementSurvey': (1, 5)}
    for col, (minimo, maximo) in rangos.items():
        invalidos = ((df[col] < minimo) | (df[col] > maximo)).sum()
        df[col] = df[col].clip(lower=minimo, upper=maximo)
        print('[2.3] {}: {} valores fuera del rango [{}, {}] ajustados al límite.'
              .format(col, invalidos, minimo, maximo))

    # --- 2.4 Detección de atípicos en el salario (IQR) -----------------------
    q1 = df['Salary'].quantile(0.25)
    q3 = df['Salary'].quantile(0.75)
    riq = q3 - q1
    limite_superior = q3 + 1.5 * riq
    atipicos_salario = (df['Salary'] > limite_superior).sum()
    print('[2.4] Salarios atípicos detectados (> {:.0f}): {} '
          '(se conservan, corresponden a cargos directivos).'
          .format(limite_superior, atipicos_salario))

    # =========================================================================
    # PASO 3 - INCONSISTENCIA (Duplicados, formatos y categorías)
    # =========================================================================
    separador('PASO 3 - TRATAMIENTO DE LA INCONSISTENCIA')

    # --- 3.1 Normalización de texto -----------------------------------------
    columnas_texto = df.select_dtypes(include='object').columns
    for col in columnas_texto:
        df[col] = df[col].astype(str).str.strip()
    print('[3.1] Espacios sobrantes eliminados en {} columnas de texto.'
          .format(len(columnas_texto)))

    # --- 3.2 Categorías escritas de distinta forma ---------------------------
    antes_hl = df['HispanicLatino'].nunique()
    df['HispanicLatino'] = df['HispanicLatino'].str.capitalize()
    print('[3.2] HispanicLatino: {} categorías unificadas a {}.'
          .format(antes_hl, df['HispanicLatino'].nunique()))
    print('      Valores finales:', sorted(df['HispanicLatino'].unique()))

    df['Department'] = df['Department'].str.replace(r'\s+', ' ', regex=True)
    print('      Department normalizado:', df['Department'].nunique(),
          'departamentos únicos.')

    # --- 3.3 Dependencia de atributos ----------------------------------------
    incoherentes = ((df['Termd'] == 1) & (df['DateofTermination'] == 'Activo')).sum()
    df.loc[(df['Termd'] == 1) & (df['DateofTermination'] == 'Activo'),
           'EmpleadoActivo'] = 0
    print('[3.3] Registros con Termd = 1 pero sin fecha de baja: {} '
          '(coherencia corregida).'.format(incoherentes))

    # --- 3.4 Registros duplicados --------------------------------------------
    duplicados_id = df.duplicated(subset=['EmpID']).sum()
    df = df.drop_duplicates(subset=['EmpID'], keep='first').copy()
    duplicados_totales = df.duplicated().sum()
    df = df.drop_duplicates().copy()
    print('[3.4] Duplicados por EmpID eliminados: {}'.format(duplicados_id))
    print('      Filas completamente duplicadas eliminadas: {}'
          .format(duplicados_totales))

    # --- 3.5 Formato único de fechas -----------------------------------------
    df['DateofHire'] = pd.to_datetime(df['DateofHire'], errors='coerce')
    df['LastPerformanceReview_Date'] = pd.to_datetime(
        df['LastPerformanceReview_Date'], errors='coerce')
    print('[3.5] Columnas de fecha convertidas a un único formato datetime.')

    # --- 3.6 Columnas sin aporte predictivo ----------------------------------
    columnas_a_eliminar = ['Employee_Name', 'Zip', 'MarriedID', 'MaritalStatusID',
                           'DeptID', 'PerfScoreID', 'EmpStatusID', 'PositionID',
                           'FromDiversityJobFairID']
    columnas_a_eliminar = [c for c in columnas_a_eliminar if c in df.columns]
    df = df.drop(columns=columnas_a_eliminar)
    print('[3.6] Columnas eliminadas por ser identificadores o redundantes: {}'
          .format(len(columnas_a_eliminar)))

    df = df.reset_index(drop=True)

    # =========================================================================
    # PASO 4 - VERIFICACIÓN DE LA LIMPIEZA
    # =========================================================================
    separador('PASO 4 - DATOS LIMPIOS CORRECTAMENTE')

    print('Valores nulos restantes por columna (solo se listan los mayores a 0):')
    nulos_finales = df.isnull().sum()
    print(nulos_finales[nulos_finales > 0] if nulos_finales.sum() > 0
          else 'Ninguno. El dataset no presenta valores nulos.')

    print('\nRegistros duplicados restantes :', df.duplicated().sum())
    print('Filas iniciales                :', filas_iniciales)
    print('Filas finales                  :', len(df))
    print('Columnas finales               :', df.shape[1])

    # Guardado del dataset limpio
    try:
        df.to_csv(ruta_salida, index=False)
        print('\nDataset limpio guardado en:', ruta_salida)
    except Exception as e:
        print('\n[Aviso] No se pudo guardar el CSV en la ruta especificada:', e)

    # =========================================================================
    # PASO 5 - PARTICIÓN EN ENTRENAMIENTO Y PRUEBA (80 / 20 ESTRATIFICADA)
    # =========================================================================
    separador('PASO 5 - RESULTADO DE LA PARTICIÓN')

    # Variable objetivo (etiqueta): Termd (1 = el empleado se fue, 0 = sigue)
    caracteristicas = ['Salary', 'EngagementSurvey', 'EmpSatisfaction',
                       'SpecialProjectsCount', 'DaysLateLast30', 'Absences',
                       'Edad', 'ManagerID']

    X = df[caracteristicas]
    y = df['Termd']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=semilla,
        stratify=y
    )

    print('Total de registros              :', len(df))
    print('Muestras de Entrenamiento (80%) :', len(X_train))
    print('Muestras de Prueba (20%)        :', len(X_test))
    print('Características utilizadas      :', len(caracteristicas))
    print('Etiqueta a predecir             : Termd')
    print('\nDistribución de la etiqueta en entrenamiento:')
    print(y_train.value_counts(normalize=True).round(3))
    print('\nDistribución de la etiqueta en prueba:')
    print(y_test.value_counts(normalize=True).round(3))

    return X_train, X_test, y_train, y_test


# Permite ejecutar el script de manera individual para probarlo
if __name__ == '__main__':
    limpiar_y_preparar_hr(
        ruta_entrada='datasets/human_resources/HRDataset_v14.csv',
        ruta_salida='datasets/human_resources/recursos_humanos_limpio.csv'
    )