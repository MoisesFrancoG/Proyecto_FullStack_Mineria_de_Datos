"""
Tarea de CLASIFICACION: predecir well_received (buena recepcion de la comunidad).

Decisiones metodologicas (defendidas en el reporte):
  · Anti-fuga: se EXCLUYEN positive_ratio, rating y rating_ordinal, porque Steam
    deriva la etiqueta de ellos. El modelo predice recepcion a partir de senales
    realmente independientes: precio, plataformas, antiguedad y popularidad
    (log de numero de resenas).
  · Desbalance (71.7% positivo): NO se reporta accuracy sola. Se priorizan
    precision/recall/F1 por clase, F1-macro, balanced accuracy y ROC-AUC / PR-AUC.
    Los modelos parametricos usan class_weight='balanced'.
  · Se comparan los CUATRO algoritmos del curso (Naive Bayes, K-NN, regresion
    logistica, arbol de decision). Se elige por F1-macro de validacion cruzada
    estratificada y se argumenta cual sirve mejor.
  · Sin fuga: el preprocesador se ajusta solo en train, dentro del Pipeline.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocessing import build_preprocessor, feature_columns, select_xy


def _candidates() -> dict[str, object]:
    """Los 4 algoritmos del curso, cada uno como preprocesador + estimador."""
    rs = config.RANDOM_STATE
    specs = {
        "NaiveBayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=rs
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=rs
        ),
    }
    return {
        name: Pipeline([("prep", build_preprocessor("classification")), ("model", est)])
        for name, est in specs.items()
    }


def _eval(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    rep = classification_report(y_test, pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(rep["accuracy"]), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, pred)), 4),
        "f1_macro": round(float(rep["macro avg"]["f1-score"]), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "por_clase": {
            "no_recibido_0": {
                "precision": round(float(rep["0"]["precision"]), 4),
                "recall": round(float(rep["0"]["recall"]), 4),
                "f1": round(float(rep["0"]["f1-score"]), 4),
                "support": int(rep["0"]["support"]),
            },
            "bien_recibido_1": {
                "precision": round(float(rep["1"]["precision"]), 4),
                "recall": round(float(rep["1"]["recall"]), 4),
                "f1": round(float(rep["1"]["f1-score"]), 4),
                "support": int(rep["1"]["support"]),
            },
        },
    }


def _plot_confusion(model, x_test, y_test, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_estimator(
        model, x_test, y_test, display_labels=["No (0)", "Si (1)"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set(title=f"Matriz de confusion — {title}")
    fig.savefig(config.FIGURES_DIR / "fig_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_roc(fitted: dict, x_test, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for name, model in fitted.items():
        proba = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set(title="Curvas ROC", xlabel="FPR", ylabel="TPR")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(config.FIGURES_DIR / "fig_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def train() -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    x, y = select_xy(df, "classification")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, model in _candidates().items():
        cv_f1 = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = {
            "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
            "cv_f1_macro_std": round(float(cv_f1.std()), 4),
            "test": _eval(model, x_test, y_test),
        }
        t = results[name]["test"]
        print(f"[clf] {name:18} CV-F1m={cv_f1.mean():.3f}  test F1m={t['f1_macro']:.3f}  "
              f"ROC-AUC={t['roc_auc']:.3f}  bal-acc={t['balanced_accuracy']:.3f}")

    # Seleccion por F1-macro de validacion cruzada (equilibra ambas clases).
    best_name = max(results, key=lambda k: results[k]["cv_f1_macro_mean"])
    best_model = fitted[best_name]
    print(f"[clf] modelo elegido: {best_name}")

    joblib.dump(best_model, config.CLF_MODEL_PATH)
    _plot_confusion(best_model, x_test, y_test, best_name)
    _plot_roc(fitted, x_test, y_test)

    metrics = {
        "task": "classification",
        "target": config.CLASSIFICATION_TARGET,
        "positivo_pct_train": round(100 * float(y_train.mean()), 2),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": feature_columns("classification"),
        "excluidas_por_fuga": ["positive_ratio", "rating", "rating_ordinal"],
        "modelos": results,
        "modelo_elegido": best_name,
        "metricas_test": results[best_name]["test"],
        "metrica_seleccion": "f1_macro (CV estratificada)",
        "nota_metrica": "Datos desbalanceados (71.7% positivo): se prioriza F1-macro, "
                        "balanced accuracy y ROC/PR-AUC sobre accuracy.",
    }
    config.CLF_METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[clf] modelo -> {config.CLF_MODEL_PATH.name} | metricas -> {config.CLF_METRICS_PATH.name}")
    return metrics


if __name__ == "__main__":
    train()
