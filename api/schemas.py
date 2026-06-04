"""Esquemas Pydantic: contratos de entrada/salida de la API."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Inferencia: REGRESION (precio) ──────────────────────────────────────────
class PriceRequest(BaseModel):
    release_year: int = Field(..., ge=1990, le=2025, examples=[2021])
    positive_ratio: int = Field(..., ge=0, le=100, examples=[85],
                                description="% de resenas positivas")
    user_reviews: int = Field(..., ge=0, examples=[1500])
    win: bool = True
    mac: bool = False
    linux: bool = False


class PriceResponse(BaseModel):
    predicted_price_usd: float
    model: str
    note: str = "Precio base estimado (USD) para un juego de pago."


# ── Inferencia: CLASIFICACION (recepcion) ───────────────────────────────────
class ReceptionRequest(BaseModel):
    release_year: int = Field(..., ge=1990, le=2025, examples=[2021])
    price_original: float = Field(..., ge=0, examples=[19.99])
    user_reviews: int = Field(..., ge=0, examples=[1500])
    win: bool = True
    mac: bool = False
    linux: bool = False


class ReceptionResponse(BaseModel):
    well_received: bool
    probability: float = Field(..., description="P(bien recibido)")
    label: str
    model: str


# ── OLAP ────────────────────────────────────────────────────────────────────
class CubeRequest(BaseModel):
    dimension: str = Field("year", examples=["year"])
    measure: str = Field("price_original", examples=["price_original"])
    agg: str = Field("avg", examples=["avg"])
    paid_only: bool = False
