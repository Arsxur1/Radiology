"""Request handling helpers for the API — build a signal, run the pipeline.

Kept separate from the FastAPI wiring so it is unit-testable without HTTP.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import secrets
import shutil
import sqlite3
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from cardiolens.layer1_ingestion import (
    generate_synthetic_ecg,
    load_csv,
    load_dicom,
    load_edf,
    load_wfdb,
)
from cardiolens.pipeline import PipelineResult, run_pipeline

# Guardrails for the public-facing service.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024   # 64 MB total across uploaded files
STORE_MAX_ENTRIES = 200               # bounded in-memory analysis cache

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


def _wfdb_data_files(hea: Path) -> set[str]:
    """Data files a WFDB header refers to (signal lines after the record line)."""
    files: set[str] = set()
    lines = [ln for ln in hea.read_text(encoding="latin-1").splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    for ln in lines[1:]:
        files.add(ln.split()[0])
    return files


def signal_from_uploads(
    files: list[UploadFile],
    *,
    sampling_rate_hz: float | None = None,
    unit: str = "uv",
):
    """Persist uploaded files to a temp dir and dispatch to a parser.

    Supports WFDB (.hea + .dat), EDF (.edf/.bdf), DICOM-ECG (.dcm) and CSV (.csv/.txt).
    """
    if not files:
        raise ValueError("no files uploaded")
    total = sum(len(f.content) for f in files)
    if total > MAX_UPLOAD_BYTES:
        raise ValueError(f"upload too large ({total} bytes > {MAX_UPLOAD_BYTES})")
    tmp = Path(tempfile.mkdtemp(prefix="cardiolens_api_"))
    try:
        saved: list[Path] = []
        for f in files:
            # Path(...).name strips any directory components (traversal-safe).
            p = tmp / Path(f.filename).name
            p.write_bytes(f.content)
            saved.append(p)

        hea = next((p for p in saved if p.suffix.lower() == ".hea"), None)
        if hea is not None:
            # A header may only point at files uploaded with it (D104): no other path on
            # the server is ever read, whatever the WFDB library would accept.
            uploaded = {p.name for p in saved}
            # Multi-segment headers name segment records without an extension.
            missing = sorted(f for f in _wfdb_data_files(hea)
                             if f not in uploaded and f + ".hea" not in uploaded)
            if missing:
                raise ValueError(f"WFDB header refers to files that were not uploaded: {missing}")
            return load_wfdb(hea.with_suffix(""))
        edf = next((p for p in saved if p.suffix.lower() in (".edf", ".bdf")), None)
        if edf is not None:
            return load_edf(edf)
        dcm = next((p for p in saved if p.suffix.lower() == ".dcm"), None)
        if dcm is not None:
            return load_dicom(dcm)
        csv = next((p for p in saved if p.suffix.lower() in (".csv", ".txt")), None)
        if csv is not None:
            if sampling_rate_hz is None:
                raise ValueError("sampling_rate_hz is required for CSV input")
            return load_csv(csv, sampling_rate_hz=sampling_rate_hz, unit=unit)
        raise ValueError("unsupported upload; expected WFDB(.hea+.dat), EDF, DICOM-ECG (.dcm) or CSV")
    except ValueError:
        raise
    except Exception as exc:  # a malformed file is a client error, not a 500 (D104)
        raise ValueError(f"could not parse the uploaded file: {type(exc).__name__}") from exc
    finally:
        # Parsers read the raw signal into memory, so the temp files are no
        # longer needed once parsing returns (or raises) — never leak them.
        shutil.rmtree(tmp, ignore_errors=True)


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


# --- Bounded in-memory analysis store (MVP; not for multi-worker production) ---
# LRU-evicting so a long-running server can't grow without bound. Swap for a DB
# / object store in production (see DECISIONS.md).
_STORE: OrderedDict[str, PipelineResult] = OrderedDict()


def _db() -> tuple[sqlite3.Connection, bytes] | None:
    """Opt-in persistent store (D109): CARDIOLENS_DB_PATH + CARDIOLENS_DB_KEY.

    Results hold numpy arrays, so they are pickled; every record carries an HMAC of
    the bytes under the key and is unpickled only if it verifies, so a tampered or
    foreign database file is never deserialised. Unset path — in-memory store only.
    """
    path = os.environ.get("CARDIOLENS_DB_PATH")
    if not path:
        return None
    key = os.environ.get("CARDIOLENS_DB_KEY", "")
    if len(key) < 16:
        raise RuntimeError("CARDIOLENS_DB_PATH needs CARDIOLENS_DB_KEY of at least 16 characters")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, created REAL, mac TEXT, blob BLOB)")
    return con, key.encode()


def _mac(key: bytes, analysis_id: str, blob: bytes) -> str:
    return hmac.new(key, analysis_id.encode() + b"\0" + blob, hashlib.sha256).hexdigest()


def _persist(analysis_id: str, result: PipelineResult) -> None:
    db = _db()
    if db is None:
        return
    con, key = db
    blob = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    now = time.time()
    with con:
        con.execute("INSERT OR REPLACE INTO analyses VALUES (?, ?, ?, ?)",
                    (analysis_id, now, _mac(key, analysis_id, blob), blob))
        days = float(os.environ.get("CARDIOLENS_DB_RETENTION_DAYS", "0") or 0)
        if days > 0:  # opt-in retention (D110): older records are purged on each write
            con.execute("DELETE FROM analyses WHERE created < ?", (now - days * 86400.0,))
    con.close()


def delete_analysis(analysis_id: str) -> bool:
    """Remove an analysis from memory and from the persistent store (D110)."""
    found = _STORE.pop(analysis_id, None) is not None
    db = _db()
    if db is not None:
        con, _ = db
        with con:
            found = con.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,)).rowcount > 0 or found
        con.close()
    return found


def _load(analysis_id: str) -> PipelineResult | None:
    db = _db()
    if db is None:
        return None
    con, key = db
    row = con.execute("SELECT mac, blob FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    con.close()
    if row is None or not hmac.compare_digest(row[0], _mac(key, analysis_id, row[1])):
        return None
    return pickle.loads(row[1])  # bytes verified by the HMAC above


def run_and_store(sig, *, mode: str | None,
                  prior_id: str | None = None) -> tuple[str, PipelineResult]:
    prior = get_analysis(prior_id) if prior_id else None
    if prior_id:
        sig.context.prior_ecg_id = prior_id
    result = run_pipeline(sig, mode=mode, prior=prior)
    analysis_id = secrets.token_hex(16)   # 128 bits: analyses carry age, sex and findings (D104)
    _STORE[analysis_id] = result
    _STORE.move_to_end(analysis_id)
    while len(_STORE) > STORE_MAX_ENTRIES:
        _STORE.popitem(last=False)  # evict least-recently-used
    _persist(analysis_id, result)
    return analysis_id, result


_HITS: dict[str, list[float]] = {}


def rate_limited(client: str, *, now: float | None = None) -> bool:
    """Sliding one-minute window per client for the analysis endpoint (D104).

    ``CARDIOLENS_RATE_LIMIT_PER_MIN`` sets the limit (default 60; 0 turns it off).
    In-process only, like the analysis store — a production deployment needs a shared one.
    """
    limit = int(os.environ.get("CARDIOLENS_RATE_LIMIT_PER_MIN", "60"))
    if limit <= 0:
        return False
    now = time.monotonic() if now is None else now
    hits = [t for t in _HITS.get(client, []) if now - t < 60.0]
    if len(hits) >= limit:
        _HITS[client] = hits
        return True
    hits.append(now)
    _HITS[client] = hits
    if len(_HITS) > 10_000:                      # bound memory: drop idle clients
        for k in [k for k, v in _HITS.items() if not v or now - v[-1] >= 60.0]:
            del _HITS[k]
    return False


def get_analysis(analysis_id: str) -> PipelineResult | None:
    result = _STORE.get(analysis_id)
    if result is not None:
        _STORE.move_to_end(analysis_id)  # mark as recently used
        return result
    result = _load(analysis_id)            # after a restart or LRU eviction (D109)
    if result is not None:
        _STORE[analysis_id] = result
        while len(_STORE) > STORE_MAX_ENTRIES:
            _STORE.popitem(last=False)
    return result
