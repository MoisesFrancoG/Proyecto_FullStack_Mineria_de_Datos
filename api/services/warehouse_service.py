"""
Servicio OLAP sobre el warehouse DuckDB (esquema estrella).

Todas las consultas van contra la vista denormalizada `v_games` (fact + 3 dims).
La conexion se abre en modo SOLO LECTURA: la API nunca modifica el warehouse, y
read_only permite varias conexiones concurrentes sin bloqueos.

Seguridad: el endpoint de "cubo" generico permite elegir dimension/medida/agg,
pero SOLO desde listas blancas que mapean a columnas/SQL reales. Nunca se
interpola texto del usuario en el SQL -> sin inyeccion.
"""
from __future__ import annotations

from functools import lru_cache

import duckdb

from ..config import WAREHOUSE_DB

# ── Listas blancas: nombre publico -> expresion SQL segura ───────────────────
DIMENSIONS: dict[str, str] = {
    "year": "year",
    "decade": "decade",
    "quarter": "quarter",
    "sentiment_bucket": "sentiment_bucket",
    "rating_label": "rating_label",
    "platform_combo": "platform_combo",
    "num_platforms": "num_platforms",
    "is_free": "is_free",
}

MEASURES: dict[str, str] = {
    "count": "*",
    "price_original": "price_original",
    "price_final": "price_final",
    "positive_ratio": "positive_ratio",
    "user_reviews": "user_reviews",
    "well_received": "well_received",
    "discount": "discount",
}

AGGS: dict[str, str] = {
    "avg": "AVG", "median": "MEDIAN", "sum": "SUM",
    "min": "MIN", "max": "MAX", "count": "COUNT",
}


def _connect() -> duckdb.DuckDBPyConnection:
    if not WAREHOUSE_DB.exists():
        raise FileNotFoundError(
            f"No existe el warehouse en {WAREHOUSE_DB}. "
            "Ejecuta primero el pipeline: python -m ml.src.run_pipeline"
        )
    return duckdb.connect(str(WAREHOUSE_DB), read_only=True)


def kpis() -> dict:
    """Tarjetas resumen para la portada del frontend."""
    sql = """
        SELECT
            COUNT(*)                                        AS total_juegos,
            SUM(CASE WHEN price_original > 0 THEN 1 ELSE 0 END) AS juegos_pago,
            ROUND(AVG(CASE WHEN price_original > 0 THEN price_original END), 2) AS precio_medio_pago,
            ROUND(100.0 * AVG(well_received), 1)            AS pct_bien_recibido,
            ROUND(AVG(positive_ratio), 1)                   AS positive_ratio_medio,
            ROUND(100.0 * AVG(is_free), 1)                  AS pct_gratis
        FROM v_games
    """
    with _connect() as con:
        return con.execute(sql).df().iloc[0].to_dict()


def cube(dimension: str, measure: str, agg: str, paid_only: bool = False) -> list[dict]:
    """Cubo OLAP generico: agrega `measure` por `dimension` con `agg`.

    Equivale a un roll-up/slice configurable desde la app.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension invalida: {dimension}. Opciones: {list(DIMENSIONS)}")
    if measure not in MEASURES:
        raise ValueError(f"medida invalida: {measure}. Opciones: {list(MEASURES)}")
    if agg not in AGGS:
        raise ValueError(f"agregacion invalida: {agg}. Opciones: {list(AGGS)}")

    dim_sql = DIMENSIONS[dimension]
    if measure == "count":
        value_sql = "COUNT(*)"
    else:
        value_sql = f"{AGGS[agg]}({MEASURES[measure]})"

    where = "WHERE price_original > 0" if paid_only else ""
    sql = f"""
        SELECT {dim_sql} AS dimension,
               ROUND(CAST({value_sql} AS DOUBLE), 3) AS value,
               COUNT(*) AS n
        FROM v_games
        {where}
        GROUP BY {dim_sql}
        ORDER BY {dim_sql}
    """
    with _connect() as con:
        df = con.execute(sql).df()
    df["dimension"] = df["dimension"].astype(str)
    return df.to_dict(orient="records")


def price_by_year(platform_combo: str | None = None) -> list[dict]:
    """Roll-up temporal: precio base medio por anio (juegos de pago)."""
    params: list = []
    where = "WHERE price_original > 0"
    if platform_combo:
        where += " AND platform_combo = ?"
        params.append(platform_combo)
    sql = f"""
        SELECT year, ROUND(AVG(price_original), 2) AS precio_medio,
               ROUND(MEDIAN(price_original), 2) AS precio_mediano, COUNT(*) AS n
        FROM v_games {where}
        GROUP BY year ORDER BY year
    """
    with _connect() as con:
        return con.execute(sql, params).df().to_dict(orient="records")


def reception_by_platform() -> list[dict]:
    """Slice por plataforma: tasa de buena recepcion y precio medio."""
    sql = """
        SELECT platform_combo,
               COUNT(*) AS n,
               ROUND(100.0 * AVG(well_received), 1) AS pct_bien_recibido,
               ROUND(AVG(positive_ratio), 1) AS positive_ratio_medio,
               ROUND(AVG(CASE WHEN price_original > 0 THEN price_original END), 2) AS precio_medio
        FROM v_games
        GROUP BY platform_combo
        ORDER BY n DESC
    """
    with _connect() as con:
        return con.execute(sql).df().to_dict(orient="records")


def rating_distribution() -> list[dict]:
    """Distribucion por etiqueta de rating (ordenada por ranking ordinal)."""
    sql = """
        SELECT rating_label, ordinal_rank, sentiment_bucket, COUNT(*) AS n
        FROM v_games
        GROUP BY rating_label, ordinal_rank, sentiment_bucket
        ORDER BY ordinal_rank
    """
    with _connect() as con:
        return con.execute(sql).df().to_dict(orient="records")


def top_games(by: str = "user_reviews", limit: int = 20, paid_only: bool = False) -> list[dict]:
    """Drill-down al detalle: top juegos por una medida."""
    if by not in MEASURES or by == "count":
        by = "user_reviews"
    where = "WHERE price_original > 0" if paid_only else ""
    sql = f"""
        SELECT title, year, platform_combo, rating_label,
               positive_ratio, user_reviews, price_original
        FROM v_games {where}
        ORDER BY {MEASURES[by]} DESC
        LIMIT ?
    """
    with _connect() as con:
        return con.execute(sql, [int(limit)]).df().to_dict(orient="records")


@lru_cache(maxsize=1)
def dimension_options() -> dict:
    """Valores disponibles de las dimensiones (para poblar selects del frontend)."""
    with _connect() as con:
        combos = con.execute(
            "SELECT DISTINCT platform_combo FROM v_games ORDER BY platform_combo"
        ).df()["platform_combo"].tolist()
        yr = con.execute("SELECT MIN(year) lo, MAX(year) hi FROM v_games").df().iloc[0]
    return {
        "dimensions": list(DIMENSIONS),
        "measures": list(MEASURES),
        "aggs": list(AGGS),
        "platform_combos": combos,
        "year_min": int(yr["lo"]),
        "year_max": int(yr["hi"]),
    }
