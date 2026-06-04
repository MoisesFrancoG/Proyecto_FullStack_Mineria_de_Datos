"""
API REST del pipeline de mineria (FastAPI).

Expone DOS familias de endpoints sobre el mismo backend real:
  · /olap/*     -> consultas dimensionales OLAP sobre el warehouse DuckDB
  · /predict/*  -> inferencia en vivo de los modelos de regresion y clasificacion

El frontend (Streamlit) consume exclusivamente esta API; no hay resultados
precocinados. Documentacion interactiva en /docs.

Arranque:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS
from .routers import olap, predict
from .services import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga modelos en memoria al arrancar (1a peticion sin latencia de IO).
    inference_service.warmup()
    yield


app = FastAPI(
    title="Steam Data Mining API",
    description="OLAP sobre warehouse DuckDB + inferencia de modelos de precio y recepcion.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(olap.router)
app.include_router(predict.router)


@app.get("/", tags=["Salud"], summary="Estado de la API")
def root() -> dict:
    return {
        "service": "Steam Data Mining API",
        "status": "ok",
        "docs": "/docs",
        "endpoints": {
            "olap": ["/olap/kpis", "/olap/cube", "/olap/price-by-year",
                     "/olap/reception-by-platform", "/olap/rating-distribution",
                     "/olap/top-games", "/olap/options"],
            "inferencia": ["/predict/price", "/predict/reception", "/predict/metrics"],
        },
    }


@app.get("/health", tags=["Salud"], summary="Healthcheck")
def health() -> dict:
    return {"status": "healthy"}
