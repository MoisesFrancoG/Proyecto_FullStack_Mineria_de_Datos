# Steam Data Mining — Pipeline Full Stack

Pipeline reproducible de minería de datos sobre el catálogo de juegos de **Steam**
(50.872 juegos, 1997–2023), del dato crudo a un producto con interfaz: warehouse
dimensional → EDA → modelos → **API REST** → **frontend**.

> Proyecto Corte 1 · Minería de Datos · UPCh 2026A.

---

## 1. Las dos preguntas

| Tarea | Pregunta | Target | Modelo elegido |
|-------|----------|--------|----------------|
| **Regresión** | ¿Qué precio base soporta un juego de pago? | `price_original` (USD) | HistGradientBoosting |
| **Clasificación** | ¿Será **bien recibido** por la comunidad? | `well_received` (binario) | DecisionTree |

El dataset es real e imperfecto (precios inconsistentes, columnas casi constantes,
variables muy sesgadas), suficientemente rico para sostener ambas preguntas a la vez.
Justificación completa en el [reporte técnico](report/reporte.md).

## 2. Arquitectura (4 capas)

```
 data/raw/games.csv
        │
        ▼
 ┌──────────────┐   limpieza + features      ┌────────────────────────┐
 │   ml/  (ETL) │ ─────────────────────────► │ data/processed/        │
 │              │                            │  · games_clean.parquet │
 │  · cleaning  │   esquema ESTRELLA         │  · warehouse.duckdb    │
 │  · warehouse │ ─────────────────────────► │    (fact + 3 dims)     │
 │  · eda       │                            └────────────────────────┘
 │  · modelado  │   modelos entrenados        ┌────────────────────────┐
 │              │ ─────────────────────────► │ ml/models/*.joblib     │
 └──────────────┘                            └────────────────────────┘
        │                                              │
        ▼                                              ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  api/  (FastAPI)   /olap/*  consultas OLAP   ·   /predict/*  ML   │
 └──────────────────────────────────────────────────────────────────┘
        │  HTTP/JSON
        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  frontend/  (Streamlit)   KPIs · Explorador OLAP · Predicciones   │
 └──────────────────────────────────────────────────────────────────┘
```

- **`ml/`** — núcleo de minería (limpieza, warehouse, EDA, modelos). *Subproyecto.*
- **`api/`** — API REST; expone OLAP e inferencia. No importa código de `ml/`,
  solo consume los artefactos (`warehouse.duckdb`, `*.joblib`). *Subproyecto.*
- **`frontend/`** — app Streamlit; consume **solo** la API (nada precocinado).

## 3. Estructura del repositorio

```
.
├── data/
│   ├── raw/games.csv               # dataset crudo (fuente)
│   └── processed/                  # GENERADO: parquet limpio + warehouse DuckDB
├── ml/                             # ── CAPA DE MINERÍA ──
│   ├── src/
│   │   ├── config.py               # rutas, semillas, features por tarea (anti-fuga)
│   │   ├── data_cleaning.py        # crudo → limpio (determinista)
│   │   ├── warehouse.py            # esquema estrella en DuckDB
│   │   ├── preprocessing.py        # ColumnTransformer parametrizado por tarea
│   │   ├── train_regression.py     # precio: Linear/RF/HistGB + variante log
│   │   ├── train_classification.py # recepción: NB/KNN/LogReg/Árbol
│   │   └── run_pipeline.py         # orquestador (1 comando)
│   ├── eda/eda.py                  # EDA reproducible (figuras + resumen JSON)
│   ├── models/                     # GENERADO: *.joblib + *_metrics.json
│   └── reports/                    # GENERADO: figures/*.png + eda_summary.json
├── api/                            # ── CAPA API (FastAPI) ──
│   ├── main.py                     # app + CORS + routers
│   ├── routers/{olap,predict}.py
│   └── services/{warehouse,inference}_service.py
├── frontend/app.py                 # ── CAPA PRESENTACIÓN (Streamlit) ──
├── report/reporte.md               # reporte técnico (→ PDF)
├── requirements.txt
├── AI_USAGE.md                     # declaración de uso de IA (obligatorio)
└── README.md
```

## 4. Reproducción (paso a paso)

Requisitos: **Python 3.12** y `git`. Los comandos están en PowerShell (Windows);
los equivalentes bash van como comentario.

### 4.1 Entorno e instalación

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # bash: source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Construir todo el pipeline de minería (1 comando)

```powershell
python -m ml.src.run_pipeline
```

Esto ejecuta, de forma determinista (semilla 42): **limpieza → warehouse →
EDA → modelo de regresión → modelo de clasificación**, y deja los artefactos en
`data/processed/`, `ml/models/` y `ml/reports/`. Tarda ~1–2 min.

> También puedes correr cada etapa por separado:
> `python -m ml.src.data_cleaning`, `python -m ml.src.warehouse`,
> `python -m ml.eda.eda`, `python -m ml.src.train_regression`,
> `python -m ml.src.train_classification`.

### 4.3 Arrancar la API (terminal 1)

```powershell
python -m uvicorn api.main:app --port 8000
```

Documentación interactiva (Swagger): <http://127.0.0.1:8000/docs>

### 4.4 Arrancar el frontend (terminal 2)

```powershell
streamlit run frontend/app.py
```

Abre <http://localhost:8501>. Si la API corre en otra URL, cámbiala en la barra lateral.

## 5. La API en breve

| Método | Endpoint | Qué hace |
|--------|----------|----------|
| GET  | `/olap/kpis` | KPIs globales del catálogo |
| GET  | `/olap/cube?dimension=&measure=&agg=&paid_only=` | **Cubo OLAP genérico** (roll-up/slice/dice) |
| GET  | `/olap/price-by-year` | Precio base medio por año |
| GET  | `/olap/reception-by-platform` | Recepción por plataforma |
| GET  | `/olap/rating-distribution` | Distribución por rating |
| GET  | `/olap/top-games?by=&limit=` | Top juegos (drill-down) |
| POST | `/predict/price` | Inferencia de **precio** (regresión) |
| POST | `/predict/reception` | Inferencia de **recepción** (clasificación) |
| GET  | `/predict/metrics` | Métricas de evaluación de ambos modelos |

Ejemplo:

```bash
curl -X POST http://127.0.0.1:8000/predict/reception -H "Content-Type: application/json" \
  -d '{"release_year":2019,"price_original":29.99,"user_reviews":800,"win":true,"mac":false,"linux":false}'
```

## 6. Resultados (resumen)

- **Regresión (precio):** HistGradientBoosting, R² ≈ 0.14, MAE ≈ $6.8 (test).
  Señal débil pero real → el precio base apenas es predecible con los metadatos
  disponibles (faltan género, alcance, estudio). Discutido con honestidad en el reporte.
- **Clasificación (recepción):** Árbol de decisión, F1-macro ≈ 0.58, ROC-AUC ≈ 0.65,
  balanced-accuracy ≈ 0.62 sobre datos **desbalanceados** (71.7 % positivo). Se reportan
  F1/precision/recall por clase, no accuracy sola. Bate claramente a Naive Bayes y KNN.

## 7. Rigor metodológico

- **Reproducibilidad:** todo desde un comando; semilla fija; sin pasos ocultos.
- **Sin fuga de datos:** el preprocesamiento se ajusta solo en *train* (dentro del
  `Pipeline`); en clasificación se **excluyen** `positive_ratio`/`rating` (definen la
  etiqueta); en regresión se excluyen `price_final`/`discount` (derivan del target).
- **Métricas justificadas** según cada problema (ver reporte).
- **Limitaciones declaradas** abiertamente (sección 6 y reporte).

## 8. Uso de IA

Ver [`AI_USAGE.md`](AI_USAGE.md): qué se generó con IA (andamiaje: API, frontend,
boilerplate) y qué es trabajo propio comprendido (decisiones de minería).
