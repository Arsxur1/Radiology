"""Request handling helpers for the API — build a signal, run the pipeline.

Kept separate from the FastAPI wiring so it is unit-testable without HTTP.
"""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from cardiolens.layer1_ingestion import (
    generate_synthetic_ecg,
    load_csv,
    load_edf,
    load_wfdb,
)
from cardiolens.pipeline import PipelineResult, run_pipeline

# Synthetic demo cases exposed by the web UI (mirrors cardiolens.demo).
DEMO_CASES: dict[str, dict] = {
    "normal": dict(heart_rate_bpm=72),
    "brady": dict(heart_rate_bpm=44),
    "tachy": dict(heart_rate_bpm=125),
    "hyperk": dict(heart_rate_bpm=72, t_amp_uv=750, potassium=6.6),
    "stemi": dict(heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]),
    "child": dict(heart_rate_bpm=95, age_years=6),
}


@dataclass
class UploadFile:
    filename: str
    content: bytes


def signal_from_case(case: str):
    if case not in DEMO_CASES:
        raise ValueError(f"unknown case '{case}'")
    return generate_synthetic_ecg(**DEMO_CASES[case])


def signal_from_uploads(
    files: list[UploadFile],
    *,
    sampling_rate_hz: float | None = None,
    unit: str = "uv",
):
    """Persist uploaded files to a temp dir and dispatch to a parser.

    Supports WFDB (.hea + .dat), EDF (.edf/.bdf), and CSV (.csv/.txt).
    """
    if not files:
        raise ValueError("no files uploaded")
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_api_"))
    saved: list[Path] = []
    for f in files:
        p = tmp / Path(f.filename).name
        p.write_bytes(f.content)
        saved.append(p)

    hea = next((p for p in saved if p.suffix.lower() == ".hea"), None)
    if hea is not None:
        return load_wfdb(hea.with_suffix(""))
    edf = next((p for p in saved if p.suffix.lower() in (".edf", ".bdf")), None)
    if edf is not None:
        return load_edf(edf)
    csv = next((p for p in saved if p.suffix.lower() in (".csv", ".txt")), None)
    if csv is not None:
        if sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for CSV input")
        return load_csv(csv, sampling_rate_hz=sampling_rate_hz, unit=unit)
    raise ValueError("unsupported upload; expected WFDB(.hea+.dat), EDF, or CSV")


def apply_patient_overrides(
    sig,
    *,
    age_years: float | None = None,
    age_days: float | None = None,
    sex: str | None = None,
    potassium: float | None = None,
    calcium: float | None = None,
):
    if age_years is not None:
        sig.patient.age_years = age_years
    if age_days is not None:
        sig.patient.age_days = age_days
    if sex is not None:
        sig.patient.sex = sex
    if potassium is not None:
        sig.context.potassium = potassium
    if calcium is not None:
        sig.context.calcium = calcium
    return sig


# --- Simple in-memory analysis store (MVP; not for multi-worker production) ---
_STORE: dict[str, PipelineResult] = {}


def run_and_store(sig, *, mode: str | None) -> tuple[str, PipelineResult]:
    result = run_pipeline(sig, mode=mode)
    analysis_id = uuid.uuid4().hex[:12]
    _STORE[analysis_id] = result
    return analysis_id, result


def get_analysis(analysis_id: str) -> PipelineResult | None:
    return _STORE.get(analysis_id)
