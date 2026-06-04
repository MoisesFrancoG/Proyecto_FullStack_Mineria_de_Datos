# Declaración de uso de IA

**Herramienta:** Claude Code (Anthropic), modelo Claude Opus.
**Modalidad de trabajo:** desarrollo asistido por IA bajo dirección del estudiante.
El estudiante eligió el dataset, definió y confirmó el planteamiento de las dos
tareas, y revisó/validó cada decisión metodológica. La IA generó principalmente
el **andamiaje** (código de plomería de API, frontend y boilerplate), que es
justamente lo que la política del curso fomenta delegar.

Declaración honesta y por componente, como exige el enunciado.

## Resumen por componente

| Componente | Generado con IA | Decisión / comprensión propia |
|---|---|---|
| **Elección del dataset** | — | **Propia.** Steam games; selección de las dos preguntas (precio / recepción). |
| **Limpieza de datos** (`data_cleaning.py`) | Implementación del código | **Propia:** qué corregir (precios inconsistentes), qué descartar (`steam_deck`), qué derivar (log de reseñas, antigüedad, target binario) y **por qué**. |
| **Diseño del warehouse** (`warehouse.py`) | SQL/DDL del esquema | **Propia:** grano del hecho, qué es medida vs dimensión, estrella vs copo de nieve, claves surrogate. |
| **EDA** (`eda/eda.py`) | Código de las figuras | **Propia:** qué preguntas explorar y qué decisión sustenta cada figura. |
| **Preprocesamiento** (`preprocessing.py`) | Código del `ColumnTransformer` | **Propia:** escalado/one-hot por tipo, y el ajuste solo en *train*. |
| **Selección de features / anti-fuga** | — | **Propia (núcleo).** Excluir `positive_ratio`/`rating` en clasificación y `price_final`/`discount` en regresión; evitar colinealidad. |
| **Modelado** (`train_*.py`) | Código de entrenamiento/CV | **Propia:** qué algoritmos comparar, qué métricas usar y cómo leerlas. |
| **Elección de métricas** | — | **Propia.** F1-macro/ROC-AUC/PR-AUC por desbalance; RMSE/MAE/R² en precio. |
| **API REST** (`api/`) | **Mayormente IA** | Revisión y comprensión de los endpoints. |
| **Frontend Streamlit** (`frontend/`) | **Mayormente IA** | Revisión y comprensión del flujo de consumo de la API. |
| **README / este archivo** | Redacción asistida | Contenido validado por el estudiante. |
| **Reporte técnico** | Estructura/redacción asistida | Argumentos y conclusiones propios. |

## Verificación

Todo el pipeline y la app se ejecutaron y validaron localmente: el warehouse
responde consultas OLAP, ambos modelos entrenan y se evalúan sin fuga, la API sirve
OLAP e inferencia, y el frontend (5 páginas) consume la API en vivo sin errores.
