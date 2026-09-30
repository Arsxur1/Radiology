"""Layer 1 — file parsers to the unified representation (TZ §4).

Working parsers for WFDB, EDF/EDF+ and CSV (the clean-digital MVP path,
TZ §17). DICOM-ECG / SCP-ECG / GE-MUSE / Philips / HL7-aECG are declared as
interfaces with explicit ``NotImplementedError`` + TODO — we never fabricate a
result (TZ §0). All outputs are in microvolts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Patient,
    Provenance,
    SignalRepresentation,
)

# Canonical spelling per uppercase key — sources spell aVR/aVL/aVF and the
# precordials in mixed case (PTB-XL loads "AVL", others "avl"). Downstream code
# and _TERRITORIES key on the canonical STANDARD_12_LEADS casing, and lead
# lookups are case-sensitive, so a mis-cased name silently drops that lead from
# axis / ST-territory / reciprocal logic. Normalise every parser's output here.
_CANONICAL_LEAD = {name.upper(): name for name in STANDARD_12_LEADS}


def canonicalize_lead_names(names: list[str]) -> list[str]:
    """Map lead names to the canonical 12-lead casing; leave unknowns untouched."""
    return [_CANONICAL_LEAD.get(n.strip().upper(), n) for n in names]


def _mv_to_uv(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) * 1000.0).astype(np.float32)


_AGE_RE = re.compile(r"age>?\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_SEX_RE = re.compile(r"sex>?\s*:\s*(male|female|m|f)\b", re.IGNORECASE)


def patient_from_comments(comments) -> Patient:
    """Age and sex from WFDB header comments (``# Age: 74``, ``# Sex: Female`` as in
    PhysioNet Challenge 2021; ``#<age>: 65 <sex>: F`` as in INCART). Missing or
    unparsable values (``NaN``, ``Unknown``) stay None."""
    text = " ".join(comments or [])
    age = _AGE_RE.search(text)
    sex = _SEX_RE.search(text)
    age_years = float(age.group(1)) if age else None
    if age_years is not None and not 0 <= age_years <= 120:
        age_years = None
    sex_norm = None
    if sex:
        sex_norm = "female" if sex.group(1).lower().startswith("f") else "male"
    return Patient(age_years=age_years, sex=sex_norm)


def load_wfdb(record_path: str | Path) -> SignalRepresentation:
    """Parse a WFDB record (``.hea``/``.dat``) — MIT-BIH, PTB-XL (TZ §4).

    ``record_path`` is the record name WITHOUT extension (wfdb convention).
    WFDB physical units are typically millivolts -> converted to µV.
    """
    import wfdb

    record_path = str(record_path)
    rec = wfdb.rdrecord(record_path)
    sig = np.asarray(rec.p_signal, dtype=np.float32).T  # -> [leads, samples]

    units = [u.lower() if u else "mv" for u in (rec.units or [])]
    # Convert per-lead to µV based on declared units.
    out = np.empty_like(sig)
    for i in range(sig.shape[0]):
        unit = units[i] if i < len(units) else "mv"
        out[i] = sig[i] * 1000.0 if unit in ("mv", "millivolt") else sig[i]

    lead_names = list(rec.sig_name) if rec.sig_name else [f"ch{i}" for i in range(sig.shape[0])]
    lead_names = canonicalize_lead_names(lead_names)
    return SignalRepresentation(
        signal=out,
        sampling_rate_hz=float(rec.fs),
        lead_names=lead_names,
        duration_s=float(sig.shape[1] / rec.fs),
        source="wfdb",
        # Demographics drive the age/sex norms (QTc limit, pediatric tables).
        patient=patient_from_comments(getattr(rec, "comments", None)),
        provenance=Provenance(digitized=False, confidence=1.0),
    )


def load_edf(path: str | Path) -> SignalRepresentation:
    """Parse an EDF/EDF+ file (TZ §4). Assumes signals in mV -> µV."""
    import pyedflib

    path = str(path)
    with pyedflib.EdfReader(path) as f:
        n = f.signals_in_file
        labels = f.getSignalLabels()
        fs = float(f.getSampleFrequency(0))
        data = [f.readSignal(i) for i in range(n)]
    min_len = min(len(d) for d in data)
    sig = np.vstack([d[:min_len] for d in data]).astype(np.float32)
    sig = _mv_to_uv(sig)
    return SignalRepresentation(
        signal=sig,
        sampling_rate_hz=fs,
        lead_names=canonicalize_lead_names(list(labels)),
        duration_s=float(min_len / fs),
        source="edf",
        provenance=Provenance(digitized=False, confidence=1.0),
    )


def load_csv(
    path: str | Path,
    *,
    sampling_rate_hz: float,
    lead_names: list[str] | None = None,
    unit: str = "uv",
    delimiter: str = ",",
    has_header: bool = True,
) -> SignalRepresentation:
    """Parse a CSV of samples (columns = leads) (TZ §4).

    Sampling rate is REQUIRED (not stored in CSV) — never guessed (TZ §0).
    ``unit`` in {'uv','mv'} controls scaling to microvolts.
    """
    path = Path(path)
    header_names: list[str] | None = None
    skip = 0
    if has_header:
        header_names = path.read_text(encoding="utf-8").splitlines()[0].split(delimiter)
        skip = 1
    arr = np.loadtxt(path, delimiter=delimiter, skiprows=skip, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    sig = arr.T  # [leads, samples]
    if unit.lower() == "mv":
        sig = _mv_to_uv(sig)
    names = lead_names or header_names or [f"ch{i}" for i in range(sig.shape[0])]
    return SignalRepresentation(
        signal=sig.astype(np.float32),
        sampling_rate_hz=float(sampling_rate_hz),
        lead_names=canonicalize_lead_names(list(names)),
        duration_s=float(sig.shape[1] / sampling_rate_hz),
        source="csv",
        provenance=Provenance(digitized=False, confidence=1.0),
    )


# --- Interfaces declared but not implemented in this increment (TZ §0, §17) ---
def load_dicom(path: str | Path) -> SignalRepresentation:  # pragma: no cover
    raise NotImplementedError("TODO(v2): DICOM-ECG Waveform parsing (TZ §4).")


def load_scp(path: str | Path) -> SignalRepresentation:  # pragma: no cover
    raise NotImplementedError("TODO(v2): SCP-ECG parsing (TZ §4).")


def load_from_image(path: str | Path, **kwargs) -> SignalRepresentation:
    """Digitise a single-lead trace from a clean gridded image (TZ §4).

    Clean scans/PDF renders only in this increment; dirty-photo robustness and
    multi-lead segmentation are v2 (TZ §17). Delegates to ``digitize_image``.
    """
    from cardiolens.layer1_ingestion.digitize import digitize_image

    return digitize_image(path, **kwargs)


def load(path: str | Path, **kwargs) -> SignalRepresentation:
    """Dispatch to a parser by file extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".hea", ".dat", ""):
        return load_wfdb(p.with_suffix(""))
    if ext in (".edf", ".bdf"):
        return load_edf(p)
    if ext in (".csv", ".txt"):
        return load_csv(p, **kwargs)
    if ext in (".dcm",):
        return load_dicom(p)
    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        return load_from_image(p, **kwargs)
    raise ValueError(f"Unsupported extension: {ext}")
