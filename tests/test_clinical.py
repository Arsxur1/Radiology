"""Clinical criteria encoded from ECG references (pericarditis, atrial
enlargement, calcium electrolyte chains). Tested at the Measurements level
(D14 pattern) — see docs/CLINICAL_REFERENCES.md for sources.
"""

import numpy as np

from cardiolens.config import load_norms, load_rule_base
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Patient,
    PatientContext,
    SignalRepresentation,
)
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer5_causal import analyze
from cardiolens.pipeline import run_pipeline


def _sig(age=40, calcium=None):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=age, sex="male"),
        context=PatientContext(calcium=calcium),
    )


def _codes(m, sig=None):
    sig = sig or _sig()
    return detect_findings(sig, m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def _st(elevated_leads):
    # Elevation is read at J+60 (see _st_findings); set both points. 250 uV clears
    # the higher V2-V3 precordial threshold (200) as well as the 100 limb one.
    v = 250
    return Morphology(st=[STMeasurement(lead=l, j_point_uv=(v if l in elevated_leads else 0),
                                        j60_uv=(v if l in elevated_leads else 0))
                          for l in STANDARD_12_LEADS])


def test_diffuse_st_elevation_is_pericarditis_not_stemi():
    # Diffuse across inferior + lateral + anterior, no reciprocal depression.
    diffuse = {"II", "aVF", "I", "aVL", "V4", "V5", "V6", "V2", "V3"}
    codes = _codes(Measurements(heart_rate_bpm=75, intervals=Intervals(qrs_ms=90),
                                morphology=_st(diffuse)))
    assert "pericarditis" in codes
    assert "st_elevation" not in codes


def test_focal_st_elevation_stays_stemi():
    codes = _codes(Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90),
                                morphology=_st({"II", "III", "aVF"})))
    assert "st_elevation" in codes
    assert "pericarditis" not in codes


def test_left_atrial_enlargement_from_p_duration():
    assert "left_atrial_enlargement" in _codes(
        Measurements(heart_rate_bpm=70, p_wave_duration_ms=140, intervals=Intervals(qrs_ms=90)))


def test_right_atrial_enlargement_from_p_amplitude():
    assert "right_atrial_enlargement" in _codes(
        Measurements(heart_rate_bpm=70, p_wave_amplitude_uv=300, intervals=Intervals(qrs_ms=90)))


def test_normal_p_wave_no_atrial_enlargement():
    codes = _codes(Measurements(heart_rate_bpm=70, p_wave_duration_ms=90,
                                p_wave_amplitude_uv=120, intervals=Intervals(qrs_ms=90)))
    assert not ({"left_atrial_enlargement", "right_atrial_enlargement"} & codes)


def test_hypocalcemia_long_qt_chain():
    sig = _sig(calcium=1.9)
    m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90, qtc_bazett_ms=500))
    fnd = detect_findings(sig, m, norms=load_norms("adult"), la_ra_reversed=False)
    g = analyze(fnd, load_rule_base(), sig.context)
    assert "long_qt" in fnd.codes()
    assert "hypocalcemia_long_qt" in g.activated_rules


def test_hypercalcemia_short_qt_chain():
    sig = _sig(calcium=2.8)
    m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90, qtc_bazett_ms=330))
    fnd = detect_findings(sig, m, norms=load_norms("adult"), la_ra_reversed=False)
    g = analyze(fnd, load_rule_base(), sig.context)
    assert "short_qt" in fnd.codes()
    assert "hypercalcemia_short_qt" in g.activated_rules


def test_pericarditis_causal_rule_activates():
    diffuse = {"II", "aVF", "I", "aVL", "V4", "V5", "V6", "V2", "V3"}
    m = Measurements(heart_rate_bpm=75, intervals=Intervals(qrs_ms=90), morphology=_st(diffuse))
    fnd = detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)
    g = analyze(fnd, load_rule_base(), _sig().context)
    assert "pericarditis_chain" in g.activated_rules


def test_normal_ecg_clean_of_new_findings():
    codes = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.codes()
    assert not ({"pericarditis", "left_atrial_enlargement", "right_atrial_enlargement"} & codes)


def test_long_qt_alert_tier_uses_fridericia_above_90_bpm():
    # D96: at 150/min a QT of 330 ms is QTc 520 (Bazett) but 446 (Fridericia): the
    # long-QT finding stays (Bazett), the §10 alert does not fire.
    from cardiolens.contracts.features import Intervals, Measurements
    from cardiolens.contracts.findings import Finding, FindingSet
    from cardiolens.safety.critical import scan_critical

    def scan(hr, qtcb, qtcf):
        m = Measurements(intervals=Intervals(qtc_bazett_ms=qtcb, qtc_fridericia_ms=qtcf))
        m.heart_rate_bpm = hr
        fs = FindingSet(findings=[Finding(code="long_qt", label_ru="x", label_uz="x", confidence=0.85)])
        return [a.code for a in scan_critical(fs, m).alerts]

    assert "long_qt" not in scan(150, 520, 446)
    assert "long_qt" in scan(150, 560, 505)
    assert "long_qt" in scan(70, 520, 505)      # at <= 90/min Bazett decides as before
