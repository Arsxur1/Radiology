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
