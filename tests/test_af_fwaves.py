"""Atrial fibrillation by fibrillatory waves (D58).

The amplitude P gate takes coarse f-waves for organised P on a third of the
PTB-XL AF records. An irregular rhythm whose QRS-blanked residual carries a high
4-9 Hz band-power index, with no conducted P, is AF by a second route.
"""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _m(rr_cv, p_ratio, fwave, p_onset=0.2, pr_std=80.0):
    return Measurements(heart_rate_bpm=90, rhythm_regular=False, rr_cv=rr_cv, p_wave_ratio=p_ratio,
                        p_onset_ratio=p_onset, pr_std_ms=pr_std, fwave_index=fwave,
                        intervals=Intervals(qrs_ms=90, qt_ms=360, qtc_bazett_ms=420))


def _codes(m):
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_fwaves_route_calls_af_when_the_p_gate_sees_organised_p():
    assert "atrial_fibrillation" in _codes(_m(0.2, 0.7, 0.6))      # P gate says organised, f-waves say AF
    assert "atrial_fibrillation" not in _codes(_m(0.2, 0.7, 0.2))  # no f-waves: the P gate stands


def test_fwaves_route_still_needs_irregularity_and_no_conducted_p():
    assert "atrial_fibrillation" not in _codes(_m(0.05, 0.7, 0.9))                       # regular
    assert "atrial_fibrillation" not in _codes(_m(0.2, 0.7, 0.9, p_onset=0.8, pr_std=10))  # conducted P


def test_fwave_index_is_measured_and_low_on_a_sinus_record():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    m = measure(sig)
    assert m.fwave_index is not None
    assert m.fwave_index < 0.45
