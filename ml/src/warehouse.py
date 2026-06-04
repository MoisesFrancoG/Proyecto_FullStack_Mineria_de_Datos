"""
Construccion del Data Warehouse dimensional (esquema ESTRELLA) sobre DuckDB.

Diseno (justificado en el reporte):
  · Grano del hecho = UN juego.  fact_game tiene una fila por app_id.
  · Medidas (hechos numericos): positive_ratio, user_reviews, precios, descuento,
    well_received... todo lo aditivo/promediable.
  · Dimensiones (contexto por el que se filtra/agrupa):
        dim_date      -> tiempo de lanzamiento (anio, trimestre, mes, decada)
        dim_rating    -> sentimiento ordinal de la comunidad
        dim_platform  -> combinacion de plataformas soportadas
  · Claves surrogate DETERMINISTAS (no aleatorias) para que el build sea
    reproducible:  date_key = yyyymmdd,  rating_key = rango 1..9,
    platform_key = win*4+mac*2+linux.

Esquema estrella (no copo de nieve) porque las dimensiones son pequenas y planas;
no hay jerarquias que justifiquen normalizar mas. Es la eleccion correcta para
consultas OLAP rapidas (roll-up / slice / dice) desde la API.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from . import config


# DDL del esquema estrella. Se construye desde el parquet limpio.
_BUILD_SQL = """
DROP TABLE IF EXISTS fact_game;
DROP TABLE IF EXISTS dim_date;
DROP TABLE IF EXISTS dim_rating;
DROP TABLE IF EXISTS dim_platform;

-- ── DIMENSION TIEMPO ────────────────────────────────────────────────────────
CREATE TABLE dim_date AS
SELECT DISTINCT
    CAST(strftime(date_release, '%Y%m%d') AS INTEGER) AS date_key,
    date_release                                       AS full_date,
    release_year                                       AS year,
    release_quarter                                    AS quarter,
    release_month                                      AS month,
    release_decade                                     AS decade
FROM clean;

-- ── DIMENSION RATING (sentimiento ordinal de la comunidad) ──────────────────
CREATE TABLE dim_rating AS
SELECT DISTINCT
    rating_ordinal      AS rating_key,
    rating              AS rating_label,
    rating_ordinal      AS ordinal_rank,
    rating_bucket       AS sentiment_bucket,
    well_received       AS is_well_received
FROM clean
ORDER BY rating_key;

-- ── DIMENSION PLATAFORMA (combinacion soportada) ────────────────────────────
CREATE TABLE dim_platform AS
SELECT DISTINCT
    (win*4 + mac*2 + linux) AS platform_key,
    win, mac, linux,
    num_platforms,
    platform_combo
FROM clean
ORDER BY platform_key;

-- ── HECHO: un juego por fila, con FKs a las 3 dimensiones ───────────────────
CREATE TABLE fact_game AS
SELECT
    app_id                                              AS game_key,   -- clave degenerada
    app_id,
    title,
    CAST(strftime(date_release, '%Y%m%d') AS INTEGER)   AS date_key,
    rating_ordinal                                      AS rating_key,
    (win*4 + mac*2 + linux)                             AS platform_key,
    -- medidas
    positive_ratio,
    user_reviews,
    log_user_reviews,
    price_original,
    price_final,
    discount,
    is_free,
    well_received,
    game_age_years
FROM clean;

-- ── VISTA denormalizada para consultas OLAP comodas desde la API ────────────
CREATE OR REPLACE VIEW v_games AS
SELECT
    f.game_key, f.app_id, f.title,
    d.year, d.quarter, d.month, d.decade, d.full_date,
    r.rating_label, r.ordinal_rank, r.sentiment_bucket, r.is_well_received,
    p.platform_combo, p.num_platforms, p.win, p.mac, p.linux,
    f.positive_ratio, f.user_reviews, f.log_user_reviews,
    f.price_original, f.price_final, f.discount, f.is_free, f.well_received,
    f.game_age_years
FROM fact_game f
JOIN dim_date     d ON f.date_key     = d.date_key
JOIN dim_rating   r ON f.rating_key   = r.rating_key
JOIN dim_platform p ON f.platform_key = p.platform_key;
"""


def build_warehouse(df: pd.DataFrame | None = None) -> None:
    """Construye (o reconstruye) el warehouse DuckDB desde el parquet limpio."""
    config.ensure_dirs()
    if df is None:
        df = pd.read_parquet(config.CLEAN_PARQUET)

    # Conexion fresca: si el .duckdb existe lo sobrescribimos para idempotencia.
    if config.WAREHOUSE_DB.exists():
        config.WAREHOUSE_DB.unlink()

    con = duckdb.connect(str(config.WAREHOUSE_DB))
    try:
        con.register("clean", df)
        con.execute(_BUILD_SQL)
        con.unregister("clean")

        # Resumen de control para el log.
        for tbl in ("dim_date", "dim_rating", "dim_platform", "fact_game"):
            n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            print(f"[warehouse] {tbl:13} -> {n:,} filas")
    finally:
        con.close()
    print(f"[warehouse] esquema estrella listo -> {config.WAREHOUSE_DB}")


if __name__ == "__main__":
    build_warehouse()
