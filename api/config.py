"""
Configuracion de la API.  La capa API es un subproyecto independiente: NO importa
codigo de `ml/`. Solo necesita saber donde estan los artefactos que el pipeline
de mineria produjo (el warehouse DuckDB y los modelos .joblib).
"""
from __future__ import annotations

import os
from pathlib import Path

# api/config.py -> parents[0]=api  parents[1]=<raiz repo>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

WAREHOUSE_DB: Path = PROJECT_ROOT / "data" / "processed" / "warehouse.duckdb"
MODELS_DIR: Path = PROJECT_ROOT / "ml" / "models"
REG_MODEL_PATH: Path = MODELS_DIR / "regression_price.joblib"
CLF_MODEL_PATH: Path = MODELS_DIR / "classification_reception.joblib"
REG_METRICS_PATH: Path = MODELS_DIR / "regression_metrics.json"
CLF_METRICS_PATH: Path = MODELS_DIR / "classification_metrics.json"

# Anio de corte del dataset (coincide con SNAPSHOT_DATE=2024-01-01 del pipeline).
# Se usa para derivar la antiguedad del juego en inferencia a partir del anio.
SNAPSHOT_YEAR: int = 2024

# Origen permitido para CORS (el frontend Streamlit). Configurable por entorno.
CORS_ORIGINS: list[str] = os.getenv(
    "API_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
).split(",")
