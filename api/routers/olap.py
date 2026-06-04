"""Router OLAP: consultas dimensionales sobre el warehouse DuckDB."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..services import warehouse_service as wh

router = APIRouter(prefix="/olap", tags=["OLAP"])


@router.get("/kpis", summary="KPIs globales del catalogo")
def get_kpis() -> dict:
    return wh.kpis()


@router.get("/options", summary="Valores disponibles para construir consultas")
def get_options() -> dict:
    return wh.dimension_options()


@router.get("/cube", summary="Cubo OLAP generico (dimension x medida x agregacion)")
def get_cube(
    dimension: str = Query("year", description="Dimension a agrupar"),
    measure: str = Query("price_original", description="Medida a agregar"),
    agg: str = Query("avg", description="Funcion de agregacion"),
    paid_only: bool = Query(False, description="Solo juegos de pago"),
) -> list[dict]:
    try:
        return wh.cube(dimension, measure, agg, paid_only)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/price-by-year", summary="Precio base medio por anio")
def get_price_by_year(platform_combo: str | None = Query(None)) -> list[dict]:
    return wh.price_by_year(platform_combo)


@router.get("/reception-by-platform", summary="Recepcion por combinacion de plataformas")
def get_reception_by_platform() -> list[dict]:
    return wh.reception_by_platform()


@router.get("/rating-distribution", summary="Distribucion por etiqueta de rating")
def get_rating_distribution() -> list[dict]:
    return wh.rating_distribution()


@router.get("/top-games", summary="Top juegos por una medida (drill-down)")
def get_top_games(
    by: str = Query("user_reviews"),
    limit: int = Query(20, ge=1, le=100),
    paid_only: bool = Query(False),
) -> list[dict]:
    return wh.top_games(by, limit, paid_only)
