"""
Pipeline de preprocesamiento PARAMETRIZADO y reutilizable.

Una sola funcion `build_preprocessor(task)` arma el ColumnTransformer adecuado a
partir de las listas de features declaradas en config. No hay transformaciones
ad hoc dispersas por los scripts: limpieza (en data_cleaning) + este transformador
son todo el preprocesamiento.

Punto critico anti-fuga: este transformador se ENVUELVE dentro del Pipeline de
sklearn y se ajusta (`fit`) exclusivamente con el split de entrenamiento. La media
y desviacion del escalado, y las categorias del one-hot, se aprenden solo de train.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def _features_for(task: str) -> tuple[list[str], list[str], list[str]]:
    if task == "regression":
        return (
            config.REGRESSION_NUMERIC,
            config.REGRESSION_CATEGORICAL,
            config.REGRESSION_BINARY,
        )
    if task == "classification":
        return (
            config.CLASSIFICATION_NUMERIC,
            config.CLASSIFICATION_CATEGORICAL,
            config.CLASSIFICATION_BINARY,
        )
    raise ValueError(f"task debe ser 'regression' o 'classification', no {task!r}")


def feature_columns(task: str) -> list[str]:
    """Lista ordenada de columnas de entrada que el modelo de `task` consume."""
    numeric, categorical, binary = _features_for(task)
    return [*numeric, *categorical, *binary]


def build_preprocessor(task: str) -> ColumnTransformer:
    """Construye el ColumnTransformer para la tarea dada.

    · numericas   -> StandardScaler (necesario para KNN, regresion logistica/lineal)
    · categoricas -> OneHotEncoder(handle_unknown='ignore') para no romper si en
                     inferencia llega una combinacion no vista
    · binarias    -> passthrough (ya son 0/1, escalar no aporta)
    """
    numeric, categorical, binary = _features_for(task)

    transformers = [("num", StandardScaler(), numeric)]
    if categorical:
        transformers.append(
            # sparse_output=False -> matriz densa, necesaria para GaussianNB.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)
        )
    if binary:
        transformers.append(("bin", "passthrough", binary))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def select_xy(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (features declaradas) e y (target) para la tarea.

    Para regresion descarta juegos gratis (price_original == 0): la pregunta de
    pricing solo tiene sentido sobre juegos de pago.
    """
    if task == "regression":
        df = df[df["price_original"] > 0].copy()
        target = config.REGRESSION_TARGET
    else:
        target = config.CLASSIFICATION_TARGET

    x = df[feature_columns(task)].copy()
    y = df[target].copy()
    return x, y
