# Arranca la API (FastAPI) y el frontend (Streamlit) en ventanas separadas.
# Requisito previo (una vez):
#   python -m venv .venv ; .\.venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
#   python -m ml.src.run_pipeline      # genera warehouse + modelos
#
# Uso:  .\run_app.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "No existe .venv. Crea el entorno e instala requirements primero (ver README)." }
if (-not (Test-Path (Join-Path $root "data\processed\warehouse.duckdb"))) {
    throw "No existe el warehouse. Ejecuta primero: python -m ml.src.run_pipeline"
}

Write-Host "Lanzando API en http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$py' -m uvicorn api.main:app --port 8000"

Start-Sleep -Seconds 2
Write-Host "Lanzando frontend en http://localhost:8501 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '$py' -m streamlit run '$root\frontend\app.py'"

Write-Host "Listo. Cierra las ventanas para detener los servicios." -ForegroundColor Cyan
