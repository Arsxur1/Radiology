"""Pediatric ST-elevation alert (D66): thresholds above the cohort's 98th percentile,
V4 with its own entry, and the early-repolarisation gate (small next to T, upsloping)."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement, TWave
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig(age):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=age, sex="male"),
    )


def _codes(age, st, t=None):
    norms = load_norms("pediatric" if age < 16 else "adult")
    m = Measurements(heart_rate_bpm=80, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     intervals=Intervals(pr_ms=140, qrs_ms=85, qt_ms=360, qtc_bazett_ms=390),
                     morphology=Morphology(
                         st=[STMeasurement(lead=lead, j_point_uv=j, j60_uv=j60, j80_uv=j60, qrs_net_uv=800) for lead, j, j60 in st],
                         t_waves=[TWave(lead=lead, amplitude_uv=a, qrs_ratio=0.5, half_width_ms=90, asymmetry=1.3) for lead, a in (t or [])]))
    return detect_findings(_sig(age), m, norms=norms, la_ra_reversed=False).codes()


def test_adolescent_v2_v4_elevation_under_the_pediatric_98th_percentile_is_no_alert():
    st = [("V2", 220, 230), ("V3", 230, 240), ("V4", 180, 190)]
    assert "st_elevation" not in _codes(13, st)          # pediatric: V2-V3 250, V4 200
    assert "st_elevation" in _codes(45, st)              # the same leads in an adult man over 40: V2-V3 200, V4 100
    assert "st_elevation_any" in _codes(13, st)          # the neutral descriptor still shows it


def test_a_real_pediatric_stemi_pattern_still_alerts():
    st = [("V2", 400, 380), ("V3", 450, 420), ("II", 20, 20)]
    assert "st_elevation" in _codes(13, st, t=[("V2", 600), ("V3", 650)])   # J/T 0.67: not early repolarisation


def test_early_repolarisation_gate_needs_small_j_next_to_t_and_an_upsloping_st():
    st = [("V2", 260, 330), ("V3", 270, 340)]
    er = _codes(12, st, t=[("V2", 1200), ("V3", 1300)])  # J/T 0.21, J+60 > J
    assert "st_elevation" not in er and "early_repolarisation" in er
    flat = _codes(12, st=[("V2", 260, 240), ("V3", 270, 250)], t=[("V2", 1200), ("V3", 1300)])
    assert "st_elevation" in flat                        # not upsloping: the alert stands
    assert "early_repolarisation" not in _codes(45, st, t=[("V2", 1200), ("V3", 1300)])  # adult table has no gate
