"""
Tarea de REGRESION: predecir price_original (precio base de un juego de pago).

Decisiones metodologicas (defendidas en el reporte):
  · Solo juegos de pago (price_original > 0): la pregunta de pricing no aplica a
    los gratis.
  · Modelamos el target en USD directamente. Probamos tambien un target
    log-transformado (skew ~5.2) pero EMPEORA el R2 en escala original
    (0.06 vs 0.14): el log subpondera los titulos caros. Lo dejamos como
    candidato "HistGB_logTarget" para que la evidencia quede en metrics.json.
  · Sin fuga: el preprocesador (escalado, one-hot) se ajusta dentro del Pipeline,
    solo con el split de entrenamiento. Se valida con K-fold sobre train.
  · Se comparan lineal (interpretable), RandomForest y HistGradientBoosting
    (captan no linealidades). Se elige por RMSE de validacion cruzada.
  · Baseline ingenuo (predecir la mediana) como piso de comparacion.
  · Honestidad: el R2 es bajo (~0.14). El dataset carece de los verdaderos
    drivers del precio (genero, alcance, estudio/editor); con los metadatos
    disponibles el precio base es solo debilmente predecible. Se discute en el reporte.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import build_preprocessor, feature_columns, select_xy


def _pipe(est) -> Pipeline:
    return Pipeline([("prep", build_preprocessor("regression")), ("model", est)])


def _candidates() -> dict[str, object]:
    """Modelos a comparar (target en USD), mas una variante con target log."""
    rs = config.RANDOM_STATE
    out: dict[str, object] = {
        "LinearRegression": _pipe(LinearRegression()),
        "RandomForest": _pipe(RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            n_jobs=-1, random_state=rs)),
        "HistGradientBoosting": _pipe(HistGradientBoostingRegressor(random_state=rs)),
    }
    # Variante con target log1p/expm1: documenta que el log empeora el R2.
    out["HistGB_logTarget"] = TransformedTargetRegressor(
        regressor=_pipe(HistGradientBoostingRegressor(random_state=rs)),
        func=np.log1p, inverse_func=np.expm1,
    )
    return out


def _eval(model, x_test, y_test) -> dict:
    pred = model.predict(x_test)
    return {
        "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_test, pred)), 4),
        "R2": round(float(r2_score(y_test, pred)), 4),
    }


def train() -> dict:
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    x, y = select_xy(df, "regression")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # Baseline ingenuo: predecir siempre la mediana de train.
    baseline_pred = np.full(len(y_test), float(np.median(y_train)))
    baseline = {
        "MAE": round(float(mean_absolute_error(y_test, baseline_pred)), 4),
        "RMSE": round(float(root_mean_squared_error(y_test, baseline_pred)), 4),
        "R2": round(float(r2_score(y_test, baseline_pred)), 4),
    }

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, model in _candidates().items():
        # Validacion cruzada en TRAIN (neg RMSE) -> seleccion sin tocar test.
        cv_rmse = -cross_val_score(
            model, x_train, y_train, cv=cv,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        model.fit(x_train, y_train)
        fitted[name] = model
        results[name] = {
            "cv_rmse_mean": round(float(cv_rmse.mean()), 4),
            "cv_rmse_std": round(float(cv_rmse.std()), 4),
            "test": _eval(model, x_test, y_test),
        }
        print(f"[reg] {name:16} CV-RMSE={cv_rmse.mean():.3f}  "
              f"test RMSE={results[name]['test']['RMSE']:.3f}  "
              f"MAE={results[name]['test']['MAE']:.3f}  R2={results[name]['test']['R2']:.3f}")

    # Seleccion por menor RMSE de validacion cruzada.
    best_name = min(results, key=lambda k: results[k]["cv_rmse_mean"])
    best_model = fitted[best_name]
    print(f"[reg] modelo elegido: {best_name}")

    joblib.dump(best_model, config.REG_MODEL_PATH)

    metrics = {
        "task": "regression",
        "target": config.REGRESSION_TARGET,
        "unidad": "USD",
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "features": feature_columns("regression"),
        "baseline_mediana": baseline,
        "modelos": results,
        "modelo_elegido": best_name,
        "metricas_test": results[best_name]["test"],
        "nota_metrica": "MAE/RMSE en USD; R2 adimensional. Target log-transformado internamente.",
    }
    config.REG_METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[reg] modelo -> {config.REG_MODEL_PATH.name} | metricas -> {config.REG_METRICS_PATH.name}")
    return metrics


if __name__ == "__main__":
    train()
