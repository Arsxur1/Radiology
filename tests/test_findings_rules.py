"""Deterministic finding-rule tests at the Measurements level (TZ §7).

Morphology-dependent findings (BBB, WPW, AF, ST territory) cannot be reliably
stimulated by the synthetic signal generator through neurokit2 delineation, so
we test the classification logic directly by constructing Measurements. These
rules are validated against real labelled data (PTB-XL) in the validation
harness / future increments.
"""

import numpy as np

from cardiolens.config import load_norms, load_rule_base
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer5_causal import analyze


def _sig(age_years=40, sex="male"):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=0.02, source="synthetic",
        patient=Patient(age_years=age_years, sex=sex),
    )


def _find(m, **over):
    norms = load_norms("adult")
    return detect_findings(_sig(**over), m, norms=norms, la_ra_reversed=False)


def test_wpw_short_pr_wide_qrs():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True,
                     intervals=Intervals(pr_ms=95, qrs_ms=130, qt_ms=380, qtc_bazett_ms=420))
    assert "wpw_preexcitation" in _find(m).codes()


def test_rbbb_from_positive_v1_terminal():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True,
                     qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=400, qtc_bazett_ms=430))
    codes = _find(m).codes()
    assert "rbbb" in codes and "lbbb" not in codes


def test_lbbb_from_negative_v1_terminal():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True,
                     qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=400, qtc_bazett_ms=430))
    codes = _find(m).codes()
    assert "lbbb" in codes and "rbbb" not in codes


def test_atrial_fibrillation():
    m = Measurements(heart_rate_bpm=95, rhythm_regular=False, rr_cv=0.28,
                     p_wave_ratio=0.1,
                     intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=360, qtc_bazett_ms=430))
    assert "atrial_fibrillation" in _find(m).codes()


def test_sinus_arrhythmia_with_conducted_p_is_not_af():
    # Irregular RR and a small P/R amplitude ratio (children), but a refined P onset
    # on most beats with a stable PR: organised atrial activity, not AF (D36).
    m = Measurements(heart_rate_bpm=75, rhythm_regular=False, rr_cv=0.13,
                     p_wave_ratio=0.0, p_onset_ratio=0.92, pr_std_ms=12.0,
                     intervals=Intervals(pr_ms=140, qrs_ms=88, qt_ms=370, qtc_bazett_ms=410))
    assert "atrial_fibrillation" not in _find(m).codes()


def test_irregular_rr_with_unstable_pr_stays_af():
    m = Measurements(heart_rate_bpm=95, rhythm_regular=False, rr_cv=0.28,
                     p_wave_ratio=0.1, p_onset_ratio=0.9, pr_std_ms=70.0,
                     intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=360, qtc_bazett_ms=430))
    assert "atrial_fibrillation" in _find(m).codes()


def test_normal_measurements_no_conduction_findings():
    m = Measurements(heart_rate_bpm=72, rhythm_regular=True, rr_cv=0.03,
                     p_wave_ratio=1.0, qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415))
    codes = _find(m).codes()
    assert not ({"rbbb", "lbbb", "wpw_preexcitation", "atrial_fibrillation"} & codes)
    assert "sinus_rhythm" in codes


def test_stemi_territory_and_reciprocal():
    # Inferior STEMI: elevation in II/III/aVF, reciprocal depression in I/aVL.
    st = []
    for lead in STANDARD_12_LEADS:
        if lead in ("II", "III", "aVF"):
            st.append(STMeasurement(lead=lead, j_point_uv=180, j60_uv=200, j80_uv=210))
        elif lead in ("I", "aVL"):
            st.append(STMeasurement(lead=lead, j_point_uv=-150, j60_uv=-160, j80_uv=-160))
        else:
            st.append(STMeasurement(lead=lead, j_point_uv=0, j60_uv=0, j80_uv=0))
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415),
                     morphology=Morphology(st=st))
    fnd = _find(m)
    stemi = fnd.by_code("st_elevation")
    assert stemi is not None
    ev = stemi.evidence_map()
    assert ev.get("territory") == "нижняя"
    assert "aVL" in ev.get("reciprocal", "") or "I" in ev.get("reciprocal", "")


def test_bbb_causal_rule_activates():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=400, qtc_bazett_ms=430))
    fnd = _find(m)
    g = analyze(fnd, load_rule_base(), _sig().context)
    assert "bbb_explains_wide_qrs" in g.activated_rules


def test_infarct_territory_rule_is_critical():
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True,
                     intervals=Intervals(pr_ms=150, qrs_ms=90),
                     morphology=Morphology(st=[
                         STMeasurement(lead=l, j_point_uv=(180 if l in ("II", "III", "aVF") else 0))
                         for l in STANDARD_12_LEADS
                     ]))
    fnd = _find(m)
    g = analyze(fnd, load_rule_base(), _sig().context)
    assert "coronary_occlusion_territory" in g.activated_rules
    assert g.critical_flags  # infarct territory rule is critical
