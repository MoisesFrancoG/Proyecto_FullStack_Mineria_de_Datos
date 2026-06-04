"""
Orquestador del pipeline de mineria completo (reproducible de un solo comando):

    python -m ml.src.run_pipeline

Ejecuta en orden:  limpieza -> warehouse (estrella) -> EDA -> regresion -> clasificacion.
Cada etapa es idempotente: reejecutar regenera los artefactos desde cero.
"""
from __future__ import annotations

import time

from . import config
from . import data_cleaning, train_classification, train_regression, warehouse


def main() -> None:
    config.ensure_dirs()
    t0 = time.perf_counter()

    print("\n=== 1/5  Limpieza ============================================")
    df = data_cleaning.build_clean_dataset(save=True)

    print("\n=== 2/5  Warehouse (esquema estrella DuckDB) =================")
    warehouse.build_warehouse(df)

    print("\n=== 3/5  EDA reproducible ===================================")
    # Import diferido: la EDA depende de matplotlib/seaborn y vive en ml/eda.
    from ml.eda.eda import run_eda
    run_eda()

    print("\n=== 4/5  Modelo de REGRESION (precio) =======================")
    train_regression.train()

    print("\n=== 5/5  Modelo de CLASIFICACION (recepcion) ================")
    train_classification.train()

    print(f"\n[pipeline] COMPLETO en {time.perf_counter() - t0:.1f}s. "
          "Artefactos en data/processed, ml/models y ml/reports.")


if __name__ == "__main__":
    main()
