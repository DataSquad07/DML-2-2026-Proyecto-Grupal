# -*- coding: utf-8 -*-

"""
MODELO GBDT - RECURSOS HUMANOS

Clasificación de rotación de personal:
Termd = 0 -> empleado continúa
Termd = 1 -> empleado se retiró
"""

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from limpieza_recursos_humanos import limpiar_y_preparar_hr


def entrenar_evaluar_gbdt():

    print("=" * 78)
    print(" MODELO GBDT - CLASIFICACIÓN DE RECURSOS HUMANOS")
    print("=" * 78)

    # Obtener datos ya limpiados y divididos
    X_train, X_test, y_train, y_test = limpiar_y_preparar_hr()

    # Crear el modelo GBDT
    modelo = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Matriz de confusión
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    print(confusion_matrix(y_test, y_pred))

    # Informe de clasificación
    print("\n--- INFORME DE CLASIFICACIÓN ---")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Sigue (0)", "Se fue (1)"]
        )
    )

    # Métricas principales
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    # Importancia de características
    importancias = modelo.feature_importances_

    plt.figure(figsize=(10, 5))
    plt.barh(X_train.columns, importancias)

    plt.xlabel("Importancia")
    plt.ylabel("Característica")
    plt.title("Importancia de características - GBDT")

    plt.tight_layout()
    plt.show()

    return modelo


if __name__ == "__main__":
    entrenar_evaluar_gbdt()