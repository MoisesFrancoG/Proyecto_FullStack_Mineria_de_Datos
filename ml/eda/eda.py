"""
EDA REPRODUCIBLE que sustenta las decisiones de modelado (no decoracion).

Genera, de forma determinista a partir de `games_clean.parquet`:
  · Figuras PNG en ml/reports/figures/
  · Un resumen cuantitativo en ml/reports/eda_summary.json

Cada figura responde a una decision concreta del pipeline:
  fig_target_price        -> el precio esta sesgado a la derecha (cola de AAA caros)
  fig_user_reviews_log    -> user_reviews necesita log (mediana 49, max 7,5 M)
  fig_rating_imbalance    -> el rating esta desbalanceado -> no usar accuracy sola
  fig_well_received       -> balance del target binario de clasificacion
  fig_corr               -> colinealidad entre numericas (justifica drops)
  fig_price_by_platform   -> el soporte multiplataforma se asocia a mayor precio
  fig_reception_by_price  -> relacion precio<->recepcion (senal para clasificacion)
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: reproducible en cualquier entorno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ml.src import config

sns.set_theme(style="whitegrid")


def _savefig(fig: plt.Figure, name: str) -> None:
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figura -> {path.name}")


def run_eda() -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    paid = df[df["price_original"] > 0]

    # 1. Distribucion del target de regresion (precio base, juegos de pago) ──────
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(paid["price_original"], bins=60, ax=ax, color="#4C72B0")
    ax.set(title="Distribucion de price_original (juegos de pago)",
           xlabel="USD", ylabel="frecuencia")
    _savefig(fig, "fig_target_price")

    # 2. user_reviews: crudo vs log1p ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["user_reviews"], bins=60, ax=axes[0], color="#C44E52")
    axes[0].set(title="user_reviews (crudo, muy sesgado)", xlabel="resenas")
    sns.histplot(df["log_user_reviews"], bins=60, ax=axes[1], color="#55A868")
    axes[1].set(title="log1p(user_reviews) (estabilizado)", xlabel="log resenas")
    _savefig(fig, "fig_user_reviews_log")

    # 3. Desbalance de la etiqueta rating (9 clases) ─────────────────────────────
    order = sorted(config.RATING_ORDINAL, key=config.RATING_ORDINAL.get)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, y="rating", order=order, ax=ax, palette="viridis", hue="rating", legend=False)
    ax.set(title="Conteo por rating (desbalanceado)", xlabel="n juegos", ylabel="")
    _savefig(fig, "fig_rating_imbalance")

    # 4. Balance del target binario well_received ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["well_received"].value_counts().sort_index()
    sns.barplot(x=["No (0)", "Si (1)"], y=counts.values, ax=ax, palette=["#C44E52", "#55A868"], hue=["No (0)", "Si (1)"], legend=False)
    ax.set(title=f"well_received  ({100*df['well_received'].mean():.1f}% positivo)", ylabel="n juegos")
    _savefig(fig, "fig_well_received")

    # 5. Correlacion entre numericas (justifica drops por colinealidad) ──────────
    num_cols = ["price_original", "price_final", "discount", "positive_ratio",
                "log_user_reviews", "num_platforms", "release_year",
                "game_age_years", "rating_ordinal"]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set(title="Correlacion de Pearson (numericas)")
    _savefig(fig, "fig_corr")

    # 6. Precio por combinacion de plataformas ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    plat_order = paid.groupby("platform_combo")["price_original"].median().sort_values().index
    sns.boxplot(data=paid, x="platform_combo", y="price_original", order=plat_order,
                ax=ax, showfliers=False, palette="crest", hue="platform_combo", legend=False)
    ax.set(title="price_original por plataformas (juegos de pago)", xlabel="", ylabel="USD")
    ax.tick_params(axis="x", rotation=30)
    _savefig(fig, "fig_price_by_platform")

    # 7. Tasa de buena recepcion por rango de precio ────────────────────────────
    tmp = df.copy()
    tmp["price_bin"] = pd.cut(tmp["price_original"],
                              bins=[-0.01, 0, 5, 10, 20, 40, 1000],
                              labels=["Gratis", "0-5", "5-10", "10-20", "20-40", "40+"])
    rate = tmp.groupby("price_bin", observed=True)["well_received"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=rate, x="price_bin", y="well_received", ax=ax, palette="mako", hue="price_bin", legend=False)
    ax.set(title="Tasa de buena recepcion por rango de precio", xlabel="precio (USD)",
           ylabel="% bien recibido")
    _savefig(fig, "fig_reception_by_price")

    # ── Resumen cuantitativo (JSON) ─────────────────────────────────────────────
    summary = {
        "n_juegos": int(len(df)),
        "n_juegos_pago": int(len(paid)),
        "rango_fechas": [str(df["date_release"].min().date()), str(df["date_release"].max().date())],
        "precio_original": {
            "media": round(float(paid["price_original"].mean()), 2),
            "mediana": round(float(paid["price_original"].median()), 2),
            "skew": round(float(paid["price_original"].skew()), 2),
            "max": round(float(paid["price_original"].max()), 2),
        },
        "user_reviews": {
            "mediana": int(df["user_reviews"].median()),
            "max": int(df["user_reviews"].max()),
            "skew_crudo": round(float(df["user_reviews"].skew()), 2),
            "skew_log": round(float(df["log_user_reviews"].skew()), 2),
        },
        "well_received_balance": {
            "positivo_pct": round(100 * float(df["well_received"].mean()), 1),
            "conteos": {str(k): int(v) for k, v in df["well_received"].value_counts().items()},
        },
        "rating_counts": {k: int(v) for k, v in df["rating"].value_counts().items()},
        "corr_release_year_vs_age": round(float(df["release_year"].corr(df["game_age_years"])), 3),
        "corr_positive_ratio_vs_rating_ordinal": round(float(df["positive_ratio"].corr(df["rating_ordinal"])), 3),
    }
    config.EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[eda] resumen -> {config.EDA_SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    s = run_eda()
    print(json.dumps(s, indent=2, ensure_ascii=False))
