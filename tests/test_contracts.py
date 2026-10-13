"""Contract validation tests (TZ §16 — explicit data contracts)."""

import numpy as np
import pytest
from pydantic import ValidationError

from cardiolens.contracts import (
    CausalGraph,
    InterpretationReport,
    SignalRepresentation,
)
from cardiolens.contracts.signal import STANDARD_12_LEADS


def _sig(n_leads=12, n_samples=5000):
    return SignalRepresentation(
        signal=np.zeros((n_leads, n_samples)),
        sampling_rate_hz=500, lead_names=STANDARD_12_LEADS[:n_leads],
        duration_s=n_samples / 500, source="synthetic",
    )


def test_signal_shape_ok():
    s = _sig()
    assert s.n_leads == 12 and s.n_samples == 5000
    assert s.lead("V5").shape == (5000,)


def test_signal_lead_count_mismatch_rejected():
    with pytest.raises(ValidationError):
        SignalRepresentation(
            signal=np.zeros((12, 100)), sampling_rate_hz=500,
            lead_names=["I", "II"], duration_s=0.2, source="synthetic",
        )


def test_signal_bad_rate_rejected():
    with pytest.raises(ValidationError):
        SignalRepresentation(
            signal=np.zeros((1, 100)), sampling_rate_hz=0,
            lead_names=["I"], duration_s=0.2, source="synthetic",
        )


def test_signal_json_roundtrip():
    d = _sig(1, 10).to_json_dict()
    assert len(d["signal"]) == 1 and len(d["signal"][0]) == 10


def test_causal_graph_export():
    g = CausalGraph()
    assert g.model_dump(by_alias=True)["nodes"] == []


def test_report_confidence_bounds():
    with pytest.raises(ValidationError):
        InterpretationReport(mode="A", confidence_value=1.5)
