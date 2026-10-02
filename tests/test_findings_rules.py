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


def test_pediatric_interval_bands_and_bbb_threshold():
    """A 12-year-old with a 100-ms QRS is inside the 9-15 y band: no bundle branch
    block; a 125-ms QRS still is one (qrs_bbb 110, D44)."""
    from cardiolens.config import norms_for_patient
    from cardiolens.contracts.signal import Patient
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer4_findings import detect_findings

    sig = generate_synthetic_ecg(age_years=12.0)
    sig.patient = Patient(age_years=12.0, sex="male")
    norms = norms_for_patient(12.0, None)
    m = Measurements(heart_rate_bpm=85, rhythm_regular=True, rr_cv=0.03, p_wave_ratio=1.0,
                     qrs_v1_terminal_positive=False, intervals=Intervals(pr_ms=150, qrs_ms=100, qt_ms=360, qtc_bazett_ms=410))
    assert not detect_findings(sig, m, norms=norms).codes() & {"lbbb", "rbbb", "wide_qrs_nonspecific"}
    m2 = m.model_copy(update={"intervals": Intervals(pr_ms=150, qrs_ms=125, qt_ms=360, qtc_bazett_ms=410)})
    assert "lbbb" in detect_findings(sig, m2, norms=norms).codes()


def test_pediatric_right_axis_does_not_raise_lead_reversal():
    from cardiolens.config import norms_for_patient
    from cardiolens.contracts.features import Axes
    from cardiolens.contracts.signal import Patient
    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer4_findings import detect_findings

    m = Measurements(heart_rate_bpm=110, rhythm_regular=True, rr_cv=0.03, p_wave_ratio=1.0,
                     intervals=Intervals(pr_ms=120, qrs_ms=70, qt_ms=300, qtc_bazett_ms=400), axes=Axes(qrs_deg=115.0))
    child = generate_synthetic_ecg(age_years=2.0); child.patient = Patient(age_years=2.0)
    assert "lead_reversal_la_ra" not in detect_findings(child, m, norms=norms_for_patient(2.0, None), la_ra_reversed=True).codes()
    adult = generate_synthetic_ecg(age_years=40.0); adult.patient = Patient(age_years=40.0)
    assert "lead_reversal_la_ra" in detect_findings(adult, m, norms=norms_for_patient(40.0, None), la_ra_reversed=True).codes()


def test_header_age_zero_is_unknown():
    from cardiolens.validation.harness import parse_patient_from_header

    assert parse_patient_from_header("#Age: 0\n#Sex: Male") == (None, "male")
    assert parse_patient_from_header("#Age: 7\n#Sex: Female") == (7.0, "female")


def test_wide_qrs_with_negative_v1_is_lbbb_only_when_definitely_wide_or_late_idt():
    # D50: an acute STEMI whose ST take-off merges with the QRS end reads
    # 118-128 ms; with a normal rS in V1 that used to be "LBBB", which silenced
    # the elevation as a block mimic. True CLBBB is >= 135 ms or has a late
    # intrinsicoid deflection in V5/V6.
    from cardiolens.contracts.features import Voltages

    def m(qrs, idt):
        return Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=False,
                            voltages=Voltages(intrinsicoid_v5_v6_ms=idt),
                            intervals=Intervals(pr_ms=160, qrs_ms=qrs, qt_ms=400, qtc_bazett_ms=430))
    assert "wide_qrs_nonspecific" in _find(m(124, 45)).codes() and "lbbb" not in _find(m(124, 45)).codes()
    assert "lbbb" in _find(m(124, 70)).codes()
    assert "lbbb" in _find(m(150, 45)).codes()
    assert "lbbb" in _find(m(150, None)).codes()
