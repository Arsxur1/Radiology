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
  POST /api/analysis/{id}/feedback   -> physician review (TZ §13)
  GET  /api/feedback/summary         -> per-finding verdicts vs shown confidence
  POST /api/calibration/fit?save=    -> fit (and optionally save) the calibrator

Core layers (1-5) run locally; only the optional Layer-6 Claude narrative
touches the cloud (TZ §12).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from starlette.background import BackgroundTask

from cardiolens.api import service
from cardiolens.calibration import TemperatureScaler, expected_calibration_error
from cardiolens.config import calibrator_path
from cardiolens.feedback import DEFAULT_FEEDBACK_PATH, Feedback, FeedbackStore
from cardiolens.outputs import (
    format_conclusion,
    render_causal_graph,
    render_waveform,
    result_to_dict,
    write_pdf,
)

app = FastAPI(title="CardioLens", version="0.1.0")

_STATIC = Path(__file__).parent / "static"


def _api_keys() -> list[str]:
    return [k.strip() for k in os.environ.get("CARDIOLENS_API_KEYS", "").split(",") if k.strip()]


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Opt-in API keys (D109): with CARDIOLENS_API_KEYS set, every /api/* call needs a
    matching X-API-Key header; the page itself and /health stay open. Unset — no check."""
    keys = _api_keys()
    given = request.headers.get("x-api-key") or ""
    if (keys and request.url.path.startswith("/api/")
            and not any(hmac.compare_digest(given, k) for k in keys)):
        response = JSONResponse({"detail": "missing or invalid API key"}, status_code=401)
        _audit(request, response.status_code, given)
        return response
    response = await call_next(request)
    _audit(request, response.status_code, given)
    return response


def _audit(request: Request, status: int, key: str) -> None:
    """Opt-in access log (D111): one JSON line per /api/* request to CARDIOLENS_AUDIT_LOG.

    Records time, method, path, status, client address and an 8-hex fingerprint of the
    API key (SHA-256 prefix) — never the key, the request body or the patient data."""
    path = os.environ.get("CARDIOLENS_AUDIT_LOG")
    if not path or not request.url.path.startswith("/api/"):
        return
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "method": request.method,
        "path": request.url.path,
        "status": status,
        "client": request.client.host if request.client else None,
        "key": hashlib.sha256(key.encode()).hexdigest()[:8] if key else None,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _cleanup(tmp_dir: str) -> BackgroundTask:
    """Delete a temp dir after the FileResponse has been sent (no disk leak)."""
    return BackgroundTask(shutil.rmtree, tmp_dir, ignore_errors=True)


def _feedback_store() -> FeedbackStore:
    return FeedbackStore.at(os.environ.get("CARDIOLENS_FEEDBACK_PATH", DEFAULT_FEEDBACK_PATH))


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
    request: Request,
    case: str | None = Form(default=None),
    mode: str | None = Form(default=None),
    sampling_rate_hz: float | None = Form(default=None),
    unit: str = Form(default="uv"),
    age_years: float | None = Form(default=None),
    age_days: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    potassium: float | None = Form(default=None),
    calcium: float | None = Form(default=None),
    prior_ecg_id: str | None = Form(default=None),
    files: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    client = request.client.host if request.client else "unknown"
    if service.rate_limited(client):
        raise HTTPException(status_code=429, detail="too many analyses; try again in a minute")
    try:
        if files:
            # Read in chunks and stop past the limit, so an oversized upload is never
            # held in memory whole (D104).
            uploads, total = [], 0
            for f in files:
                chunks = []
                while chunk := await f.read(1 << 20):
                    total += len(chunk)
                    if total > service.MAX_UPLOAD_BYTES:
                        raise ValueError(f"upload too large (> {service.MAX_UPLOAD_BYTES} bytes)")
                    chunks.append(chunk)
                uploads.append(service.UploadFile(filename=f.filename or "upload", content=b"".join(chunks)))
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
        analysis_id, result = service.run_and_store(sig, mode=mode, prior_id=prior_ecg_id)
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
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.pdf")
    write_pdf(result, path)
    return FileResponse(path, media_type="application/pdf",
                        filename=f"cardiolens_{analysis_id}.pdf",
                        background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/waveform.svg")
def waveform_svg(analysis_id: str) -> FileResponse:
    result = _require(analysis_id)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.svg")
    render_waveform(result, path)
    return FileResponse(path, media_type="image/svg+xml", background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/graph.png")
def graph_png(analysis_id: str) -> FileResponse:
    result = _require(analysis_id)
    tmp = tempfile.mkdtemp()
    path = str(Path(tmp) / f"{analysis_id}.png")
    render_causal_graph(result, path)
    return FileResponse(path, media_type="image/png", background=_cleanup(tmp))


@app.get("/api/analysis/{analysis_id}/conclusion", response_class=PlainTextResponse)
def conclusion(analysis_id: str, lang: str = "ru") -> str:
    result = _require(analysis_id)
    return format_conclusion(result, "uz" if lang == "uz" else "ru")


@app.delete("/api/analysis/{analysis_id}")
def delete_analysis(analysis_id: str) -> dict:
    """Delete one analysis on request (D110): memory, the persistent store and its physician
    feedback records (they may carry a reviewer name and a free-text comment)."""
    found = service.delete_analysis(analysis_id)
    n_feedback = _feedback_store().remove(analysis_id)
    if not found and not n_feedback:
        raise HTTPException(status_code=404, detail="analysis not found")
    return {"deleted": analysis_id, "feedback_records_removed": n_feedback}


@app.post("/api/analysis/{analysis_id}/feedback")
def submit_feedback(analysis_id: str, fb: Feedback) -> dict:
    """Capture physician corrections for future fine-tuning/calibration (TZ §13)."""
    fb.analysis_id = analysis_id
    result = service.get_analysis(analysis_id)
    if result is not None:
        if fb.mode is None:
            fb.mode = result.report.mode
        # The confidences come from the stored analysis, not the client (D87): the
        # calibrator is fitted on confidence_base, the value it is later applied to.
        by_code = {f.code: f for f in result.findings.findings}
        for ff in fb.findings:
            f = by_code.get(ff.code)
            if f is not None:
                ff.confidence_shown = f.confidence
                ff.confidence = f.confidence_base if f.confidence_base is not None else f.confidence
    _feedback_store().append(fb)
    return {"status": "recorded", "analysis_id": analysis_id,
            "n_findings": len(fb.findings)}


@app.get("/api/feedback/summary")
def feedback_summary() -> dict:
    """Per-finding physician verdicts vs the confidence shown (TZ §13, D87)."""
    return {"findings": _feedback_store().summary()}


def _admin_ok(token: str | None) -> bool:
    """Saving a calibrator changes every later analysis, so it needs the admin token (D104).

    With CARDIOLENS_ADMIN_TOKEN unset, saving is disabled.
    """
    expected = os.environ.get("CARDIOLENS_ADMIN_TOKEN")
    return bool(expected) and token is not None and hmac.compare_digest(token, expected)


@app.post("/api/calibration/fit")
def fit_calibration(save: bool = False,
                    x_cardiolens_admin_token: str | None = Header(default=None)) -> dict:
    """Fit the confidence calibrator from accumulated feedback (TZ §13, §16).

    With ``save=true`` (and the admin token, D104) a fitted scaler that lowers the
    calibration error is written where the pipeline loads it; the D85/D86 caps stay a
    ceiling either way (D87).
    """
    if save and not _admin_ok(x_cardiolens_admin_token):
        raise HTTPException(status_code=403, detail="saving the calibrator requires the admin token")
    samples = _feedback_store().calibration_dataset()
    scaler = TemperatureScaler().fit(samples)
    ece_before = expected_calibration_error(samples)
    ece_after = expected_calibration_error([(scaler.apply(c), y) for c, y in samples])
    saved = False
    if save and scaler.fitted and ece_after < ece_before:
        path = calibrator_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler.save(path)
        saved = True
    return {
        "n_samples": len(samples),
        "fitted": scaler.fitted,
        "temperature": round(scaler.temperature, 4),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "saved": saved,
    }
