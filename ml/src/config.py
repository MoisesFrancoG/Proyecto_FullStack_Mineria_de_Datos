"""
Configuracion central del pipeline de mineria de datos.

Centralizar rutas, semillas y la definicion de features en un solo lugar es
lo que hace el pipeline REPRODUCIBLE y PARAMETRIZADO (requisito del documento):
ningun script define rutas o hiperparametros "a mano", todos los importan de aqui.
"""
from __future__ import annotations

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Rutas (resueltas relativas a la raiz del repo, no al CWD -> portable)
#   ml/src/config.py  ->  parents[0]=src  parents[1]=ml  parents[2]=<raiz repo>
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_RAW_CSV: Path = PROJECT_ROOT / "data" / "raw" / "games.csv"
DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
CLEAN_PARQUET: Path = DATA_PROCESSED_DIR / "games_clean.parquet"
WAREHOUSE_DB: Path = DATA_PROCESSED_DIR / "warehouse.duckdb"

MODELS_DIR: Path = PROJECT_ROOT / "ml" / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "ml" / "reports" / "figures"

REG_MODEL_PATH: Path = MODELS_DIR / "regression_price.joblib"
CLF_MODEL_PATH: Path = MODELS_DIR / "classification_reception.joblib"
REG_METRICS_PATH: Path = MODELS_DIR / "regression_metrics.json"
CLF_METRICS_PATH: Path = MODELS_DIR / "classification_metrics.json"
EDA_SUMMARY_PATH: Path = PROJECT_ROOT / "ml" / "reports" / "eda_summary.json"

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibilidad
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# Fecha de corte del dataset (ultima recoleccion en Steam, finales de 2023).
# Se fija como constante para que "antiguedad del juego" sea DETERMINISTA y no
# dependa de la fecha en que se ejecute el pipeline (reproducibilidad).
SNAPSHOT_DATE: str = "2024-01-01"

# ─────────────────────────────────────────────────────────────────────────────
# Semantica del dominio: el "rating" de Steam es una etiqueta ordinal derivada
# del % de resenas positivas (positive_ratio) y del numero de resenas.
# La fijamos aqui una sola vez para que limpieza, warehouse y modelado coincidan.
# ─────────────────────────────────────────────────────────────────────────────
RATING_ORDINAL: dict[str, int] = {
    "Overwhelmingly Negative": 1,
    "Very Negative": 2,
    "Negative": 3,
    "Mostly Negative": 4,
    "Mixed": 5,
    "Mostly Positive": 6,
    "Positive": 7,
    "Very Positive": 8,
    "Overwhelmingly Positive": 9,
}

# Cubeta de sentimiento para OLAP (slice/dice mas legible que 9 categorias).
RATING_BUCKET: dict[str, str] = {
    "Overwhelmingly Negative": "Negativo",
    "Very Negative": "Negativo",
    "Negative": "Negativo",
    "Mostly Negative": "Negativo",
    "Mixed": "Mixto",
    "Mostly Positive": "Positivo",
    "Positive": "Positivo",
    "Very Positive": "Positivo",
    "Overwhelmingly Positive": "Positivo",
}

# Target de CLASIFICACION: "bien recibido" = familia Positive (rank >= 6).
# Mixed cae en 0 a proposito: queremos separar "claramente positivo" del resto.
WELL_RECEIVED_MIN_RANK: int = 6

# ─────────────────────────────────────────────────────────────────────────────
# Definicion de FEATURES por tarea.  Dos criterios guian la seleccion:
#
#  (A) Sin FUGA DE DATOS
#   · Regresion (target = price_original):  excluimos price_final y discount
#     porque se derivan del precio base (price_final = original*(1-discount/100)).
#     Entrenamos solo con juegos de pago (price_original > 0); por eso is_free
#     no entra (seria constante).
#   · Clasificacion (target = well_received): excluimos positive_ratio, rating y
#     rating_ordinal, porque Steam construye la etiqueta a partir de
#     positive_ratio -> usarlos seria fuga trivial (el modelo "adivinaria" el umbral).
#
#  (B) Sin REDUNDANCIA/COLINEALIDAD
#   · Las plataformas entran SOLO como categoria one-hot `platform_combo` (no
#     ademas como win/mac/linux + num_platforms, que son la misma informacion).
#   · La antiguedad entra como `game_age_years` (continuo); no usamos tambien
#     release_year, que es casi colineal.
# ─────────────────────────────────────────────────────────────────────────────
REGRESSION_TARGET: str = "price_original"
REGRESSION_NUMERIC: list[str] = ["game_age_years", "positive_ratio", "log_user_reviews"]
REGRESSION_CATEGORICAL: list[str] = ["platform_combo"]
REGRESSION_BINARY: list[str] = []

CLASSIFICATION_TARGET: str = "well_received"
CLASSIFICATION_NUMERIC: list[str] = ["game_age_years", "price_original", "log_user_reviews"]
CLASSIFICATION_CATEGORICAL: list[str] = ["platform_combo"]
CLASSIFICATION_BINARY: list[str] = ["is_free"]


def ensure_dirs() -> None:
    """Crea los directorios de salida si no existen (idempotente)."""
    for d in (DATA_PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, EDA_SUMMARY_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
