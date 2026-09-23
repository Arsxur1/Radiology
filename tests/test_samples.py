"""Real-parser path: bundled WFDB/CSV samples flow through the pipeline (TZ §4)."""

import os

import pytest

from cardiolens.layer1_ingestion import load_csv, load_wfdb
from cardiolens.pipeline import run_pipeline

_SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def test_wfdb_sample_pipeline():
    sig = load_wfdb(os.path.join(_SAMPLES, "sample_ecg"))
    assert sig.n_leads == 12
    result = run_pipeline(sig, mode="A")
    assert result.measurements.heart_rate_bpm == pytest.approx(75, abs=4)


def test_csv_sample_parses():
    sig = load_csv(os.path.join(_SAMPLES, "sample_ecg.csv"),
                   sampling_rate_hz=500, unit="uv")
    assert sig.n_leads == 12
    assert sig.lead_names[0] == "I"


def test_lead_names_canonicalized():
    # Sources spell the augmented leads in mixed case (PTB-XL WFDB loads "AVL").
    # Lead lookups and _TERRITORIES key on the canonical aVR/aVL/aVF casing, so
    # every parser must normalise — otherwise axis / ST-territory / reciprocal
    # logic silently drops those leads on real data.
    from cardiolens.layer1_ingestion.parsers import canonicalize_lead_names

    assert canonicalize_lead_names(["I", "AVR", "avl", "AvF", "v1", "V6"]) == [
        "I", "aVR", "aVL", "aVF", "V1", "V6"]
    # Unknown / non-standard labels (e.g. MIT-BIH "MLII") pass through untouched.
    assert canonicalize_lead_names(["MLII", "ch3"]) == ["MLII", "ch3"]
