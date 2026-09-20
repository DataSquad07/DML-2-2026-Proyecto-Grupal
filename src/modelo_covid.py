# -*- coding: utf-8 -*-
"""
===============================================================================
SEGUNDO AVANCE - MODELO REGRESIONAL (COVID-19)
Predicción de variables continuas mediante Regresión
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importamos la función de preparación de datos desde el script de COVID
from limpieza_covid import limpiar_y_preparar_covid


def entrenar_evaluar_regresion_covid(max_depth=5):
    print("=" * 78)
    print(" 2. MODELO REGRESIONAL - COVID-19")
    print("=" * 78)

    # 1. Obtener conjuntos de entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = limpiar_y_preparar_covid()

    # 2. Instanciar Modelos (Árbol de Regresión y Regresión Lineal)
    modelo_arbol = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    modelo_lineal = LinearRegression()

    # 3. Entrenamiento
    modelo_arbol.fit(X_train, y_train)
    modelo_lineal.fit(X_train, y_train)

    # 4. Predicciones
    y_pred_arbol = modelo_arbol.predict(X_test)
    y_pred_lineal = modelo_lineal.predict(X_test)

    # 5. Función Auxiliar de Evaluación de Regresión
    def calcular_metricas(y_real, y_pred, nombre_modelo):
        mse = mean_squared_error(y_real, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_real, y_pred)
        r2 = r2_score(y_real, y_pred)

        print(f"\n--- Métricas: {nombre_modelo} ---")
        print(f"Error Cuadrático Medio (MSE)      : {mse:.4f}")
        print(f"Raíz del Error Cuadrático (RMSE)  : {rmse:.4f}")
        print(f"Error Absoluto Medio (MAE)        : {mae:.4f}")
        print(f"Coeficiente de Determinación (R²): {r2:.4f}")
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

    metricas_arbol = calcular_metricas(y_test, y_pred_arbol, f"Árbol de Regresión (max_depth={max_depth})")
    metricas_lineal = calcular_metricas(y_test, y_pred_lineal, "Regresión Lineal")

    # 6. Gráfica Comparativa: Valores Reales vs Predichos
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred_arbol, color='blue', alpha=0.6, label='Árbol de Regresión')
    plt.scatter(y_test, y_pred_lineal, color='orange', alpha=0.6, label='Regresión Lineal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Predicción Ideal')
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title("Comparación de Predicciones - Dataset COVID-19")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return modelo_arbol, modelo_lineal


if __name__ == '__main__':
    entrenar_evaluar_regresion_covid(max_depth=5)