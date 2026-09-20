# -*- coding: utf-8 -*-
"""
PROYECTO INTEGRADOR - SEGUNDO AVANCE
Modelos Implementados:
 1. Modelos Regresionales (Linear Regression, Decision Tree Regressor, SVR)
 2. Modelos de Clasificación (Logistic Regression, Decision Tree Classifier, SVC)
"""

from src.limpieza_covid import obtener_datos_covid
from src.limpieza_hr import obtener_datos_hr

# Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC

# Evaluaciones
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

def evaluar_modelos_covid():
    print("=" * 78)
    print("1. EVALUACIÓN DE MODELOS - DATASET COVID-19 (REGRESIÓN)")
    print("=" * 78)
    
    X_train, X_test, y_train, y_test = obtener_datos_covid()

    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Árbol de Decisión (Regresor)": DecisionTreeRegressor(random_state=42),
        "Support Vector Regression (SVR)": SVR(kernel='rbf')
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        mse = mean_squared_error(y_test, predicciones)
        r2 = r2_score(y_test, predicciones)
        print(f"\n[+] {nombre}:")
        print(f"    - Error Cuadrático Medio (MSE) : {mse:.2f}")
        print(f"    - Coeficiente de Determinación (R2): {r2:.4f}")

def evaluar_modelos_hr():
    print("\n" + "=" * 78)
    print("2. EVALUACIÓN DE MODELOS - DATASET RECURSOS HUMANOS (CLASIFICACIÓN)")
    print("=" * 78)

    X_train, X_test, y_train, y_test = obtener_datos_hr()

    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Árbol de Decisión (Clasificador)": DecisionTreeClassifier(random_state=42),
        "Support Vector Classifier (SVC)": SVC(kernel='rbf')
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        predicciones = modelo.predict(X_test)
        acc = accuracy_score(y_test, predicciones)
        f1 = f1_score(y_test, predicciones, average='weighted')
        print(f"\n[+] {nombre}:")
        print(f"    - Exactitud (Accuracy) : {acc:.4f}")
        print(f"    - Puntaje F1 (F1-Score): {f1:.4f}")

if __name__ == "__main__":
    evaluar_modelos_covid()
    evaluar_modelos_hr()