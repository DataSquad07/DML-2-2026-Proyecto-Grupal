# -*- coding: utf-8 -*-
"""
===============================================================================
PROYECTO GRUPAL - MACHINE LEARNING
Punto de Entrada Principal (main.py)

Ejecución del Primer y Segundo Avance:
  1. Limpieza de datos y partición (80/20 estratificada/regresional).
  2. Entrenamiento y evaluación de modelos de Clasificación y Regresión.
===============================================================================
"""

import sys
import os

# Aseguramos que la carpeta 'src' esté en el path de búsqueda de Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from modelo_recursos_humanos import entrenar_evaluar_clasificacion_hr
from modelo_covid import entrenar_evaluar_regresion_covid


def ejecutar_proyecto():
    print("=" * 80)
    print("           EJECUCIÓN DEL PROYECTO GRUPAL DE MACHINE LEARNING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. MODELO DE CLASIFICACIÓN (RECURSOS HUMANOS)
    # -------------------------------------------------------------------------
    print("\n>>> INICIANDO PROCESAMIENTO Y MODELADO: RECURSOS HUMANOS <<<")
    try:
        modelo_hr = entrenar_evaluar_clasificacion_hr(max_depth=4, criterion='gini')
        print("\n[OK] Modelo de Clasificación de Recursos Humanos ejecutado con éxito.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema en Recursos Humanos: {e}")

    print("\n" + "-" * 80)

    # -------------------------------------------------------------------------
    # 2. MODELO REGRESIONAL (COVID-19)
    # -------------------------------------------------------------------------
    print("\n>>> INICIANDO PROCESAMIENTO Y MODELADO: COVID-19 <<<")
    try:
        modelo_arbol_covid, modelo_lineal_covid = entrenar_evaluar_regresion_covid(max_depth=5)
        print("\n[OK] Modelos de Regresión de COVID-19 ejecutados con éxito.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un problema en COVID-19: {e}")

    print("\n" + "=" * 80)
    print("           EJECUCIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 80)


if __name__ == '__main__':
    ejecutar_proyecto()