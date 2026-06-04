"""
Limpieza y construccion de features del dataset de juegos de Steam.

Esta etapa es deliberadamente DETERMINISTA y esta separada del modelado: produce
un unico `games_clean.parquet` que alimenta tanto al warehouse como a los modelos.
Cada decision de limpieza esta documentada en linea y justificada en el reporte.

Hallazgos del EDA que motivan las decisiones (ver ml/eda):
  · steam_deck es casi constante (50.870/50.872 True)  -> se descarta (no informa).
  · 985 filas con price_final > price_original             -> inconsistencia, se corrige.
  · user_reviews muy sesgada (mediana 49, max 7,5 M)       -> log1p para modelar.
  · rating es ordinal y deriva de positive_ratio           -> se mapea a rango ordinal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw() -> pd.DataFrame:
    """Carga el CSV crudo tal cual."""
    return pd.read_csv(config.DATA_RAW_CSV)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la cadena de limpieza y devuelve un DataFrame analitico.

    Es una funcion pura (no toca disco) para poder testearla y reutilizarla.
    """
    df = df.copy()

    # 1. Duplicados exactos por clave de negocio (app_id es la PK de Steam).
    df = df.drop_duplicates(subset="app_id").reset_index(drop=True)

    # 2. Tipado de fechas. errors='coerce' -> fechas invalidas quedan NaT y se quitan.
    df["date_release"] = pd.to_datetime(df["date_release"], errors="coerce")
    df = df.dropna(subset=["date_release"]).reset_index(drop=True)

    # 3. Columna casi constante: steam_deck (True en 99.996%). No aporta varianza.
    df = df.drop(columns=["steam_deck"], errors="ignore")

    # 4. Booleanos de plataforma -> enteros 0/1 (mas comodos para modelar y OLAP).
    for col in ("win", "mac", "linux"):
        df[col] = df[col].astype(int)

    # 5. Inconsistencia de precios: por definicion price_final <= price_original.
    #    En 985 filas no se cumple (ruido de la fuente). Se corrige el final al
    #    base y se recalcula el descuento de forma coherente.
    bad_price = df["price_final"] > df["price_original"]
    df.loc[bad_price, "price_final"] = df.loc[bad_price, "price_original"]

    #    Recalculamos discount real (%) a partir de precios; evita descuentos
    #    inconsistentes de la fuente. Juegos gratis -> descuento 0.
    base = df["price_original"].replace(0, np.nan)
    df["discount"] = (
        ((df["price_original"] - df["price_final"]) / base * 100)
        .fillna(0)
        .clip(lower=0, upper=100)
        .round(2)
    )

    # 6. Features derivadas -------------------------------------------------------
    snapshot = pd.Timestamp(config.SNAPSHOT_DATE)
    df["release_year"] = df["date_release"].dt.year
    df["release_month"] = df["date_release"].dt.month
    df["release_quarter"] = df["date_release"].dt.quarter
    df["release_decade"] = (df["release_year"] // 10 * 10).astype(int)
    df["game_age_years"] = ((snapshot - df["date_release"]).dt.days / 365.25).round(3)

    # Plataformas: conteo y combinacion legible (dimension OLAP + feature).
    df["num_platforms"] = df[["win", "mac", "linux"]].sum(axis=1)
    df["platform_combo"] = df.apply(_platform_combo, axis=1)

    # user_reviews fuertemente sesgada -> log1p estabiliza la escala.
    df["log_user_reviews"] = np.log1p(df["user_reviews"])

    # Precio: bandera de gratuito (free-to-play) util para OLAP y clasificacion.
    df["is_free"] = (df["price_original"] == 0).astype(int)

    # 7. Semantica del rating (ordinal + cubeta + target binario) -----------------
    df["rating_ordinal"] = df["rating"].map(config.RATING_ORDINAL).astype("Int64")
    df["rating_bucket"] = df["rating"].map(config.RATING_BUCKET)
    #    Filas con rating no reconocido (no deberia haber) se descartan.
    df = df.dropna(subset=["rating_ordinal"]).reset_index(drop=True)
    df["rating_ordinal"] = df["rating_ordinal"].astype(int)

    #    Target de clasificacion: bien recibido = rank >= 6 (familia Positive).
    df["well_received"] = (df["rating_ordinal"] >= config.WELL_RECEIVED_MIN_RANK).astype(int)

    return df


def _platform_combo(row: pd.Series) -> str:
    """Etiqueta legible de la combinacion de plataformas (dimension OLAP)."""
    tags = [name for name in ("win", "mac", "linux") if row[name] == 1]
    if not tags:
        return "Ninguna"
    pretty = {"win": "Windows", "mac": "Mac", "linux": "Linux"}
    return " + ".join(pretty[t] for t in tags)


def build_clean_dataset(save: bool = True) -> pd.DataFrame:
    """Pipeline completo crudo -> limpio. Persiste parquet si save=True."""
    config.ensure_dirs()
    df = clean(load_raw())
    if save:
        df.to_parquet(config.CLEAN_PARQUET, index=False)
        print(f"[clean] {len(df):,} filas limpias -> {config.CLEAN_PARQUET}")
    return df


if __name__ == "__main__":
    build_clean_dataset(save=True)
