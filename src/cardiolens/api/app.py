"""FastAPI service exposing the CardioLens pipeline (TZ §14).

Endpoints:
  GET  /                         -> web UI
  GET  /health                   -> liveness
  GET  /api/cases                -> available synthetic demo cases
  POST /api/analyze              -> run pipeline (synthetic case or file upload)
  GET  /api/analysis/{id}/report.pdf
  GET  /api/analysis/{id}/waveform.svg
  GET  /api/analysis/{id}/graph.png
  GET  /api/analysis/{id}/conclusion?lang=ru|uz

Core layers (1-5) run locally; only the optional Layer-6 Claude narrative
touches the cloud (TZ §12).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from cardiolens.api import service
from cardiolens.outputs import (
    format_conclusion,
    render_causal_graph,
    render_waveform,
    result_to_dict,
    write_pdf,
)

app = FastAPI(title="CardioLens", version="0.1.0")

_STATIC = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (_STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "cardiolens", "version": "0.1.0"}


@app.get("/api/cases")
def cases() -> dict:
    return {"cases": sorted(service.DEMO_CASES)}


@app.post("/api/analyze")
async def analyze(
    case: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    sampling_rate_hz: float | None = Form(default=None),
    unit: str = Form(default="uv"),
    age_years: float | None = Form(default=None),
    age_days: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    potassium: float | None = Form(default=None),
    calcium: float | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    try:
        if files:
            uploads = [
                service.UploadFile(filename=f.filename or "upload", content=await f.read())
                for f in files
            ]
            sig = service.signal_from_uploads(
                uploads, sampling_rate_hz=sampling_rate_hz, unit=unit
            )
        elif case:
            sig = service.signal_from_case(case)
        else:
            raise ValueError("provide a 'case' or upload 'files'")

        service.apply_patient_overrides(
            sig, age_years=age_years, age_days=age_days, sex=sex,
            potassium=potassium, calcium=calcium,
        )
        analysis_id, result = service.run_and_store(sig, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = result_to_dict(result)
    payload["analysis_id"] = analysis_id
    payload["conclusion_ru"] = format_conclusion(result, "ru")
    payload["conclusion_uz"] = format_conclusion(result, "uz")
    return JSONResponse(payload)


def _require(analysis_id: str):
    result = service.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="analysis not found")
    return result


@app.get("/api/analysis/{analysis_id}/report.pdf")
def report_pdf(analysis_id: str) -> FileResponse:
    result = _require(analysis_id)
    path = str(Path(tempfile.mkdtemp()) / f"{analysis_id}.pdf")
    write_pdf(result, path)
    return FileResponse(path, media_type="application/pdf",
                        filename=f"cardiolens_{analysis_id}.pdf")


@app.get("/api/analysis/{analysis_id}/waveform.svg")
def waveform_svg(analysis_id: str) -> FileResponse:
    result = _require(analysis_id)
    path = str(Path(tempfile.mkdtemp()) / f"{analysis_id}.svg")
    render_waveform(result, path)
    return FileResponse(path, media_type="image/svg+xml")


@app.get("/api/analysis/{analysis_id}/graph.png")
def graph_png(analysis_id: str) -> FileResponse:
    result = _require(analysis_id)
    path = str(Path(tempfile.mkdtemp()) / f"{analysis_id}.png")
    render_causal_graph(result, path)
    return FileResponse(path, media_type="image/png")


@app.get("/api/analysis/{analysis_id}/conclusion", response_class=PlainTextResponse)
def conclusion(analysis_id: str, lang: str = "ru") -> str:
    result = _require(analysis_id)
    return format_conclusion(result, "uz" if lang == "uz" else "ru")
