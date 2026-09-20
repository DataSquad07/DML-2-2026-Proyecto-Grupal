# -*- coding: utf-8 -*-
"""
===============================================================================
SEGUNDO AVANCE - MODELO DE CLASIFICACIÓN (RECURSOS HUMANOS)
Árbol de Decisión para predecir rotación laboral (Termd: 1 = Se fue, 0 = Sigue)
===============================================================================
"""

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Importamos la función de limpieza y partición del script de Recursos Humanos
from limpieza_recursos_humanos import limpiar_y_preparar_hr


def entrenar_evaluar_clasificacion_hr(max_depth=4, criterion='gini'):
    print("=" * 78)
    print(" 1. MODELO DE CLASIFICACIÓN - RECURSOS HUMANOS (ROTACIÓN DE PERSONAL)")
    print("=" * 78)

    # 1. Obtener conjuntos de entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = limpiar_y_preparar_hr()

    # 2. Instanciar y Entrenar el Modelo
    modelo = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # 3. Predicción sobre el conjunto de prueba
    y_pred = modelo.predict(X_test)

    # 4. Evaluación del Modelo
    print("\n--- MATRIZ DE CONFUSIÓN ---")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("\n--- INFORME DE CLASIFICACIÓN (Métricas Principales) ---")
    print(classification_report(y_test, y_pred, target_names=['Sigue (0)', 'Se Fue (1)']))

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    # 5. Visualización del Árbol de Decisión
    plt.figure(figsize=(16, 8))
    plot_tree(
        modelo,
        feature_names=X_train.columns,
        class_names=['Sigue (0)', 'Se Fue (1)'],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.title(f"Árbol de Decisión - Clasificación RRHH (max_depth={max_depth}, criterion='{criterion}')")
    plt.tight_layout()
    plt.show()

    return modelo


if __name__ == '__main__':
    entrenar_evaluar_clasificacion_hr(max_depth=4, criterion='gini')