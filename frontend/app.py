"""
Frontend Streamlit — Steam Data Mining.

Es un PRODUCTO que consume la API REST real (FastAPI): toda cifra y grafica sale
de una peticion HTTP en vivo al backend; no hay datos ni imagenes precocinadas.

Arranque:  streamlit run frontend/app.py
(requiere la API levantada;  uvicorn api.main:app --port 8000)
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_DEFAULT = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Steam Data Mining", page_icon="🎮", layout="wide")


# ── Cliente HTTP ─────────────────────────────────────────────────────────────
def api_get(path: str, params: dict | None = None):
    r = requests.get(f"{st.session_state.api_base}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict):
    r = requests.post(f"{st.session_state.api_base}{path}", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=120, show_spinner=False)
def cached_get(base: str, path: str, params: tuple | None = None):
    """Cache por (base, path, params). params como tupla para ser hasheable."""
    p = dict(params) if params else None
    r = requests.get(f"{base}{path}", params=p, timeout=30)
    r.raise_for_status()
    return r.json()


# ── Estado / sidebar ─────────────────────────────────────────────────────────
if "api_base" not in st.session_state:
    st.session_state.api_base = API_DEFAULT

st.sidebar.title("🎮 Steam Data Mining")
st.session_state.api_base = st.sidebar.text_input("URL de la API", st.session_state.api_base)

# Chequeo de conexion.
try:
    api_get("/health")
    st.sidebar.success("API conectada")
    api_ok = True
except Exception as e:
    st.sidebar.error(f"API no disponible.\n{e}")
    st.sidebar.info("Levanta el backend:\n`uvicorn api.main:app --port 8000`")
    api_ok = False

page = st.sidebar.radio(
    "Navegacion",
    ["Portada / KPIs", "Explorador OLAP", "Predecir precio",
     "Predecir recepcion", "Modelos y metricas"],
)
st.sidebar.caption("Datos servidos en vivo por la API · DuckDB + scikit-learn")

if not api_ok:
    st.title("Steam Data Mining")
    st.warning("Conecta la API desde la barra lateral para usar la aplicacion.")
    st.stop()


# ── Pagina 1: Portada / KPIs ────────────────────────────────────────────────
def page_kpis():
    st.title("Catalogo de juegos de Steam — visión general")
    k = cached_get(st.session_state.api_base, "/olap/kpis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Juegos totales", f"{int(k['total_juegos']):,}")
    c2.metric("Juegos de pago", f"{int(k['juegos_pago']):,}")
    c3.metric("Precio medio (pago)", f"${k['precio_medio_pago']}")
    c4.metric("% bien recibidos", f"{k['pct_bien_recibido']}%")
    c1, c2 = st.columns(2)
    c1.metric("Positive ratio medio", f"{k['positive_ratio_medio']}%")
    c2.metric("% gratuitos", f"{k['pct_gratis']}%")

    st.subheader("Distribución por rating de la comunidad")
    dist = pd.DataFrame(cached_get(st.session_state.api_base, "/olap/rating-distribution"))
    fig = px.bar(dist, x="n", y="rating_label", color="sentiment_bucket",
                 orientation="h", labels={"n": "n juegos", "rating_label": ""},
                 category_orders={"rating_label": dist["rating_label"].tolist()})
    fig.update_layout(height=420, legend_title="Sentimiento")
    st.plotly_chart(fig, use_container_width=True)


# ── Pagina 2: Explorador OLAP ───────────────────────────────────────────────
def page_olap():
    st.title("Explorador OLAP")
    st.caption("Construye consultas dimensionales sobre el warehouse (esquema estrella en DuckDB).")
    opts = cached_get(st.session_state.api_base, "/olap/options")

    st.subheader("Cubo configurable")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    dimension = c1.selectbox("Dimensión (agrupar por)", opts["dimensions"])
    measure = c2.selectbox("Medida", opts["measures"],
                           index=opts["measures"].index("price_original"))
    agg = c3.selectbox("Agregación", opts["aggs"])
    paid_only = c4.checkbox("Solo de pago", value=False)

    data = api_get("/olap/cube", {"dimension": dimension, "measure": measure,
                                  "agg": agg, "paid_only": paid_only})
    df = pd.DataFrame(data)
    if df.empty:
        st.info("Sin resultados.")
    else:
        label = "n juegos" if measure == "count" else f"{agg}({measure})"
        chart = "line" if dimension in ("year", "quarter") else "bar"
        fig = (px.line if chart == "line" else px.bar)(
            df, x="dimension", y="value", labels={"dimension": dimension, "value": label})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.rename(columns={"value": label}), use_container_width=True, hide_index=True)

    st.divider()
    cL, cR = st.columns(2)
    with cL:
        st.subheader("Precio base medio por año")
        pby = pd.DataFrame(cached_get(st.session_state.api_base, "/olap/price-by-year"))
        fig = px.line(pby, x="year", y=["precio_medio", "precio_mediano"],
                      labels={"value": "USD", "variable": ""})
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        st.subheader("Recepción por plataforma")
        rbp = pd.DataFrame(cached_get(st.session_state.api_base, "/olap/reception-by-platform"))
        fig = px.bar(rbp, x="platform_combo", y="pct_bien_recibido",
                     color="precio_medio", labels={"pct_bien_recibido": "% bien recibido",
                                                    "platform_combo": ""})
        fig.update_layout(height=360, xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top juegos (drill-down)")
    by = st.selectbox("Ordenar por", ["user_reviews", "positive_ratio", "price_original"])
    top = pd.DataFrame(api_get("/olap/top-games", {"by": by, "limit": 20}))
    st.dataframe(top, use_container_width=True, hide_index=True)


# ── Pagina 3: Predecir precio (regresion) ───────────────────────────────────
def page_price():
    st.title("Predecir precio base (regresión)")
    st.caption("El modelo estima el precio de catálogo (USD) de un juego de pago.")
    with st.form("price_form"):
        c1, c2, c3 = st.columns(3)
        year = c1.number_input("Año de lanzamiento", 1997, 2025, 2021)
        pos = c2.slider("Positive ratio (%)", 0, 100, 85)
        reviews = c3.number_input("Nº de reseñas", 0, 10_000_000, 1500, step=100)
        st.write("Plataformas soportadas:")
        p1, p2, p3 = st.columns(3)
        win = p1.checkbox("Windows", value=True)
        mac = p2.checkbox("Mac", value=False)
        linux = p3.checkbox("Linux", value=False)
        submitted = st.form_submit_button("Estimar precio", type="primary")
    if submitted:
        out = api_post("/predict/price", {"release_year": year, "positive_ratio": pos,
                                          "user_reviews": reviews, "win": win,
                                          "mac": mac, "linux": linux})
        st.success(f"### Precio base estimado: ${out['predicted_price_usd']}")
        st.caption(f"Modelo: {out['model']}")


# ── Pagina 4: Predecir recepcion (clasificacion) ────────────────────────────
def page_reception():
    st.title("Predecir recepción de la comunidad (clasificación)")
    st.caption("Probabilidad de que el juego sea «bien recibido» (rating positivo). "
               "El modelo NO usa el % de reseñas positivas: lo predice desde precio, "
               "plataformas, antigüedad y popularidad.")
    with st.form("recep_form"):
        c1, c2, c3 = st.columns(3)
        year = c1.number_input("Año de lanzamiento", 1997, 2025, 2019)
        price = c2.number_input("Precio base (USD)", 0.0, 300.0, 19.99, step=1.0)
        reviews = c3.number_input("Nº de reseñas", 0, 10_000_000, 800, step=100)
        p1, p2, p3 = st.columns(3)
        win = p1.checkbox("Windows", value=True)
        mac = p2.checkbox("Mac", value=False)
        linux = p3.checkbox("Linux", value=False)
        submitted = st.form_submit_button("Predecir recepción", type="primary")
    if submitted:
        out = api_post("/predict/reception", {"release_year": year, "price_original": price,
                                              "user_reviews": reviews, "win": win,
                                              "mac": mac, "linux": linux})
        prob = out["probability"]
        c1, c2 = st.columns([1, 2])
        c1.metric("¿Bien recibido?", "Sí ✅" if out["well_received"] else "No ❓")
        c1.metric("Probabilidad", f"{prob*100:.1f}%")
        fig = px.bar(x=[prob, 1 - prob], y=["Bien recibido", "No / incierto"],
                     orientation="h", range_x=[0, 1],
                     color=["Bien recibido", "No / incierto"],
                     color_discrete_sequence=["#2ca02c", "#d62728"])
        fig.update_layout(height=200, showlegend=False, xaxis_title="probabilidad")
        c2.plotly_chart(fig, use_container_width=True)
        st.caption(f"Modelo: {out['model']} · umbral 0.5")


# ── Pagina 5: Modelos y metricas ────────────────────────────────────────────
def page_models():
    st.title("Modelos y métricas de evaluación")
    m = cached_get(st.session_state.api_base, "/predict/metrics")

    st.header("Regresión — precio base")
    reg = m["regression"]
    st.write(f"**Modelo elegido:** {reg['modelo_elegido']} · "
             f"target `{reg['target']}` ({reg['unidad']}) · "
             f"n_train={reg['n_train']:,}, n_test={reg['n_test']:,}")
    reg_rows = [{"modelo": k, "CV-RMSE": v["cv_rmse_mean"], **v["test"]}
                for k, v in reg["modelos"].items()]
    reg_df = pd.DataFrame(reg_rows)
    c1, c2 = st.columns([1.4, 1])
    c1.dataframe(reg_df, use_container_width=True, hide_index=True)
    fig = px.bar(reg_df, x="modelo", y="R2", title="R² por modelo (test)")
    fig.update_layout(height=320, xaxis_tickangle=-20)
    c2.plotly_chart(fig, use_container_width=True)
    st.caption(f"Baseline (mediana): R²={reg['baseline_mediana']['R2']}, "
               f"MAE=${reg['baseline_mediana']['MAE']}. {reg['nota_metrica']}")

    st.divider()
    st.header("Clasificación — recepción")
    clf = m["classification"]
    st.write(f"**Modelo elegido:** {clf['modelo_elegido']} · target `{clf['target']}` · "
             f"{clf['positivo_pct_train']}% positivo en train")
    st.caption(f"Excluidas por fuga: {', '.join(clf['excluidas_por_fuga'])}. {clf['nota_metrica']}")
    clf_rows = [{"modelo": k, "CV-F1m": v["cv_f1_macro_mean"],
                 "F1-macro": v["test"]["f1_macro"], "ROC-AUC": v["test"]["roc_auc"],
                 "PR-AUC": v["test"]["pr_auc"], "bal-acc": v["test"]["balanced_accuracy"],
                 "accuracy": v["test"]["accuracy"]}
                for k, v in clf["modelos"].items()]
    clf_df = pd.DataFrame(clf_rows)
    c1, c2 = st.columns([1.5, 1])
    c1.dataframe(clf_df, use_container_width=True, hide_index=True)
    fig = px.bar(clf_df.melt(id_vars="modelo", value_vars=["F1-macro", "ROC-AUC", "bal-acc"]),
                 x="modelo", y="value", color="variable", barmode="group",
                 title="Comparativa de clasificadores (test)")
    fig.update_layout(height=320, xaxis_tickangle=-20, legend_title="")
    c2.plotly_chart(fig, use_container_width=True)

    best = clf["metricas_test"]["por_clase"]
    st.subheader(f"Detalle por clase — {clf['modelo_elegido']}")
    st.dataframe(pd.DataFrame(best).T, use_container_width=True)


PAGES = {
    "Portada / KPIs": page_kpis,
    "Explorador OLAP": page_olap,
    "Predecir precio": page_price,
    "Predecir recepcion": page_reception,
    "Modelos y metricas": page_models,
}
PAGES[page]()
