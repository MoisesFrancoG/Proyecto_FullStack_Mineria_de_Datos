"""
Genera report/reporte.pdf a partir de report/reporte.md.

    python report/build_pdf.py

Usa markdown -> HTML -> PDF (xhtml2pdf, puro Python, sin dependencias nativas).
Las imagenes referenciadas con rutas relativas se resuelven contra la carpeta
report/ mediante link_callback.
"""
from __future__ import annotations

from pathlib import Path

import markdown
from xhtml2pdf import pisa

HERE = Path(__file__).resolve().parent
MD = HERE / "reporte.md"
PDF = HERE / "reporte.pdf"

CSS = """
@page { size: A4; margin: 2cm 1.8cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.4; }
h1 { font-size: 19pt; color: #14532d; margin: 0 0 2pt 0; }
h2 { font-size: 14pt; color: #166534; border-bottom: 1.5px solid #16a34a; padding-bottom: 3px; margin-top: 16px; }
h3 { font-size: 11.5pt; color: #15803d; margin-top: 12px; }
p, li { text-align: justify; }
em { color: #444; }
code { font-family: Courier, monospace; background: #f1f5f9; font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; }
th, td { border: 0.6px solid #94a3b8; padding: 3px 5px; }
th { background: #dcfce7; color: #14532d; text-align: left; }
img { width: 430px; margin: 6px 0; }
blockquote { color: #475569; border-left: 3px solid #cbd5e1; margin: 6px 0; padding-left: 10px; }
"""


def link_callback(uri: str, rel: str) -> str:
    """Resuelve rutas de imagenes relativas (../ml/...) a rutas absolutas."""
    p = (HERE / uri).resolve()
    return str(p) if p.exists() else uri


def main() -> None:
    html_body = markdown.markdown(
        MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html_body}</body></html>"
    with PDF.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, link_callback=link_callback, encoding="utf-8")
    if result.err:
        raise SystemExit(f"Error generando PDF ({result.err} errores)")
    print(f"[pdf] generado -> {PDF}  ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
