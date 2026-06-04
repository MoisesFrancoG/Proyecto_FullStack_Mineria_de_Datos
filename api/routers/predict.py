"""Router de inferencia: regresion (precio) y clasificacion (recepcion)."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import (
    PriceRequest,
    PriceResponse,
    ReceptionRequest,
    ReceptionResponse,
)
from ..services import inference_service as inf

router = APIRouter(prefix="/predict", tags=["Inferencia"])


@router.post("/price", response_model=PriceResponse, summary="Estima el precio base (regresion)")
def predict_price(req: PriceRequest) -> PriceResponse:
    out = inf.predict_price(
        release_year=req.release_year,
        positive_ratio=req.positive_ratio,
        user_reviews=req.user_reviews,
        win=req.win, mac=req.mac, linux=req.linux,
    )
    return PriceResponse(**out)


@router.post("/reception", response_model=ReceptionResponse, summary="Predice la recepcion (clasificacion)")
def predict_reception(req: ReceptionRequest) -> ReceptionResponse:
    out = inf.predict_reception(
        release_year=req.release_year,
        price_original=req.price_original,
        user_reviews=req.user_reviews,
        win=req.win, mac=req.mac, linux=req.linux,
    )
    return ReceptionResponse(**out)


@router.get("/metrics", summary="Metricas de evaluacion de ambos modelos")
def get_metrics() -> dict:
    return inf.metrics()
