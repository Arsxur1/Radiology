"""Peaked ("tented") T waves of hyperkalaemia: tall, narrow and symmetric (D61).

The old rule (T > 400 µV and T/QRS > 0.5 in two leads, hard-coded) fired on 21 %
of the 664 set and 14 % of PTB-XL; half of those calls were QRS remnants inside
a fixed 100-450 ms window that reached the next beat at fast rates.
"""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology, TWave
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure import measure
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _codes_for(t_waves):
    m = Measurements(heart_rate_bpm=72, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     intervals=Intervals(qrs_ms=90, qt_ms=380, qtc_bazett_ms=400),
                     morphology=Morphology(t_waves=t_waves))
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def _t(lead, amp, ratio=1.0, hw=80.0, asym=1.0):
    return TWave(lead=lead, amplitude_uv=amp, qrs_ratio=ratio, half_width_ms=hw, asymmetry=asym)


def test_tall_narrow_symmetric_t_in_two_leads_is_the_hyperkalaemia_pattern():
    assert "peaked_t_waves" in _codes_for([_t("V3", 1200), _t("V4", 1100)])
    assert "peaked_t_waves" in _codes_for([_t("II", 600), _t("aVF", 550)])   # limb threshold 500


def test_tall_but_broad_or_asymmetric_t_is_a_normal_tall_t():
    assert "peaked_t_waves" not in _codes_for([_t("V3", 1400, hw=140), _t("V4", 1300, hw=150)])
    assert "peaked_t_waves" not in _codes_for([_t("V3", 1400, asym=2.5), _t("V4", 1300, asym=2.2)])
    assert "peaked_t_waves" not in _codes_for([_t("V3", 700), _t("V4", 650)])   # under the precordial floor
    assert "peaked_t_waves" not in _codes_for([_t("V3", 1200), _t("aVR", 1200)])  # one lead + aVR


def test_a_narrow_spike_in_the_t_window_is_not_a_t_wave():
    # the measurement leaves half-width None for sub-40 ms deflections; the rule skips them
    assert "peaked_t_waves" not in _codes_for([TWave(lead="V3", amplitude_uv=1500, qrs_ratio=1.0),
                                               TWave(lead="V4", amplitude_uv=1500, qrs_ratio=1.0)])


def test_synthetic_hyperkalaemia_is_found_and_a_fast_normal_rhythm_is_not():
    hyper = measure(generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6))
    fs = detect_findings(generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.6), hyper,
                         norms=load_norms("adult"), la_ra_reversed=False)
    assert "peaked_t_waves" in fs.codes()
    assert all(t.half_width_ms is None or t.half_width_ms >= 40 for t in hyper.morphology.t_waves)
    fast = generate_synthetic_ecg(heart_rate_bpm=130, t_amp_uv=300)
    m = measure(fast)
    assert "peaked_t_waves" not in detect_findings(fast, m, norms=load_norms("adult"), la_ra_reversed=False).codes()
    # at 130 bpm the T window must not reach the next QRS: every T descriptor is a real T
    assert all(t.half_width_ms is None or t.half_width_ms >= 40 for t in m.morphology.t_waves)
