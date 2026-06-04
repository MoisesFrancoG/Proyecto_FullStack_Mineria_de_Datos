"""
Servicio de inferencia en vivo.

Carga los Pipelines de sklearn entrenados (.joblib) UNA sola vez y construye, a
partir de las entradas del usuario, exactamente las mismas features que vieron los
modelos en entrenamiento. Replicar aqui esa derivacion (no importar de `ml/`)
mantiene la API como subproyecto autonomo; la semantica esta documentada y es
identica a ml/src/data_cleaning.py.
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from ..config import (
    CLF_METRICS_PATH,
    CLF_MODEL_PATH,
    REG_METRICS_PATH,
    REG_MODEL_PATH,
    SNAPSHOT_YEAR,
)

_PRETTY = {"win": "Windows", "mac": "Mac", "linux": "Linux"}


def _platform_combo(win: bool, mac: bool, linux: bool) -> str:
    """Misma etiqueta que en el pipeline (data_cleaning._platform_combo)."""
    tags = [name for name, on in (("win", win), ("mac", mac), ("linux", linux)) if on]
    return " + ".join(_PRETTY[t] for t in tags) if tags else "Ninguna"


def _game_age(release_year: int) -> float:
    """Antiguedad aproximada en anios respecto al corte del dataset (2024-01-01)."""
    return float(max(0, SNAPSHOT_YEAR - release_year))


@lru_cache(maxsize=1)
def _reg_model():
    return joblib.load(REG_MODEL_PATH)


@lru_cache(maxsize=1)
def _clf_model():
    return joblib.load(CLF_MODEL_PATH)


@lru_cache(maxsize=1)
def metrics() -> dict:
    """Metricas de ambos modelos (para el endpoint /predict/metrics y el frontend)."""
    return {
        "regression": json.loads(REG_METRICS_PATH.read_text(encoding="utf-8")),
        "classification": json.loads(CLF_METRICS_PATH.read_text(encoding="utf-8")),
    }


def predict_price(release_year: int, positive_ratio: int, user_reviews: int,
                  win: bool, mac: bool, linux: bool) -> dict:
    """Regresion: precio base estimado (USD) para un juego de pago."""
    row = pd.DataFrame([{
        "game_age_years": _game_age(release_year),
        "positive_ratio": float(positive_ratio),
        "log_user_reviews": float(np.log1p(user_reviews)),
        "platform_combo": _platform_combo(win, mac, linux),
    }])
    pred = float(_reg_model().predict(row)[0])
    pred = max(0.0, round(pred, 2))  # el precio no puede ser negativo
    model_name = json.loads(REG_METRICS_PATH.read_text(encoding="utf-8"))["modelo_elegido"]
    return {"predicted_price_usd": pred, "model": model_name}


def predict_reception(release_year: int, price_original: float, user_reviews: int,
                      win: bool, mac: bool, linux: bool) -> dict:
    """Clasificacion: probabilidad de buena recepcion de la comunidad."""
    row = pd.DataFrame([{
        "game_age_years": _game_age(release_year),
        "price_original": float(price_original),
        "log_user_reviews": float(np.log1p(user_reviews)),
        "platform_combo": _platform_combo(win, mac, linux),
        "is_free": int(price_original == 0),
    }])
    model = _clf_model()
    proba = float(model.predict_proba(row)[0, 1])
    well = bool(proba >= 0.5)
    model_name = json.loads(CLF_METRICS_PATH.read_text(encoding="utf-8"))["modelo_elegido"]
    return {
        "well_received": well,
        "probability": round(proba, 4),
        "label": "Bien recibido" if well else "Recepcion incierta/negativa",
        "model": model_name,
    }


def warmup() -> None:
    """Carga modelos y metricas al arrancar (evita latencia en la 1a peticion)."""
    _reg_model()
    _clf_model()
    metrics()
