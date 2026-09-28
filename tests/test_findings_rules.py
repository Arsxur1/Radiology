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


def test_first_degree_av_block_single_lead_fallback():
    # No per-lead PR data -> falls back to the single measured PR > 200.
    m = Measurements(heart_rate_bpm=70,
                     intervals=Intervals(pr_ms=240, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415))
    assert "first_degree_av_block" in _find(m).codes()


def test_first_degree_av_block_multilead_agreement():
    # Lead II alone under-measures (190), but >=2 leads agree PR > 200 -> block.
    m = Measurements(heart_rate_bpm=70,
                     intervals=Intervals(pr_ms=190, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415),
                     pr_ms_by_lead={"II": 190, "V1": 212, "V2": 208})
    assert "first_degree_av_block" in _find(m).codes()


def test_no_first_degree_av_block_when_only_one_lead_prolonged():
    # A single prolonged lead (likely artefact) with normal II must NOT trigger.
    m = Measurements(heart_rate_bpm=70,
                     intervals=Intervals(pr_ms=180, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415),
                     pr_ms_by_lead={"II": 180, "V1": 215, "V2": 175})
    assert "first_degree_av_block" not in _find(m).codes()


def test_rvh_dominant_r_in_v1():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_rs_ratio=5.0,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415))
    assert "rvh" in _find(m).codes()


def test_no_rvh_when_v1_r_not_dominant():
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_rs_ratio=0.2,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415))
    assert "rvh" not in _find(m).codes()


def test_rvh_excluded_when_rbbb():
    # A dominant V1 R from RBBB (wide QRS + positive V1 terminal) must not be RVH.
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_rs_ratio=5.0,
                     qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=150, qrs_ms=140, qt_ms=400, qtc_bazett_ms=430))
    codes = _find(m).codes()
    assert "rbbb" in codes and "rvh" not in codes


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


def test_borderline_wide_qrs_is_nonspecific_not_complete_block():
    # 110–120 ms is a nonspecific/incomplete conduction delay, not a complete
    # bundle-branch block — the biggest false-positive source on real records.
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True,
                     qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=160, qrs_ms=114, qt_ms=400, qtc_bazett_ms=430))
    codes = _find(m).codes()
    assert "lbbb" not in codes and "rbbb" not in codes
    assert "wide_qrs_nonspecific" in codes


def test_atrial_fibrillation():
    m = Measurements(heart_rate_bpm=95, rhythm_regular=False, rr_cv=0.28,
                     p_wave_ratio=0.1,
                     intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=360, qtc_bazett_ms=430))
    assert "atrial_fibrillation" in _find(m).codes()


def test_atrial_fibrillation_detected_despite_present_p_ratio():
    # Automated delineation labels a "P" for nearly every beat even in AF, so
    # p_wave_ratio stays high on real records. Detection must ride the irregular-
    # irregular RR pattern, not a hard P-absence gate (regression: this used to
    # silently suppress every real-data AF, giving 0.0 sensitivity on PTB-XL).
    m = Measurements(heart_rate_bpm=95, rhythm_regular=False, rr_cv=0.28,
                     p_wave_ratio=0.95,
                     intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=360, qtc_bazett_ms=430))
    assert "atrial_fibrillation" in _find(m).codes()


def test_sinus_with_ectopy_is_not_atrial_fibrillation():
    # A couple of premature beats inflate the std-based CV, but most RR stay
    # clustered at the median (low robust dispersion) -> not AF (D15z).
    m = Measurements(heart_rate_bpm=80, rhythm_regular=False, rr_cv=0.22, rr_mad_cv=0.02,
                     p_wave_ratio=0.95,
                     intervals=Intervals(pr_ms=160, qrs_ms=90, qt_ms=380, qtc_bazett_ms=420))
    assert "atrial_fibrillation" not in _find(m).codes()


def test_af_with_moderate_cv_but_broad_dispersion():
    m = Measurements(heart_rate_bpm=88, rhythm_regular=False, rr_cv=0.13, rr_mad_cv=0.11,
                     p_wave_ratio=0.9,
                     intervals=Intervals(pr_ms=None, qrs_ms=90, qt_ms=360, qtc_bazett_ms=430))
    assert "atrial_fibrillation" in _find(m).codes()


def test_regular_rhythm_is_not_atrial_fibrillation():
    m = Measurements(heart_rate_bpm=72, rhythm_regular=True, rr_cv=0.30,
                     p_wave_ratio=0.1,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=380, qtc_bazett_ms=415))
    assert "atrial_fibrillation" not in _find(m).codes()


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
                         STMeasurement(lead=l,
                                       j_point_uv=(180 if l in ("II", "III", "aVF") else 0),
                                       j60_uv=(180 if l in ("II", "III", "aVF") else 0))
                         for l in STANDARD_12_LEADS
                     ]))
    fnd = _find(m)
    g = analyze(fnd, load_rule_base(), _sig().context)
    assert "coronary_occlusion_territory" in g.activated_rules
    assert g.critical_flags  # infarct territory rule is critical


def _irbbb_codes(qrs, term):
    m = Measurements(heart_rate_bpm=72, rhythm_regular=True, qrs_v1_terminal_positive=term,
                     intervals=Intervals(pr_ms=160, qrs_ms=qrs, qt_ms=380, qtc_bazett_ms=415))
    return _find(m).codes()


def test_incomplete_rbbb_boundaries():
    assert "incomplete_rbbb" in _irbbb_codes(105, True)
    # 110-119 ms with an R' in V1 is incomplete RBBB, not a nonspecific delay.
    codes = _irbbb_codes(115, True)
    assert "incomplete_rbbb" in codes and "wide_qrs_nonspecific" not in codes
    assert "rbbb" in _irbbb_codes(130, True) and "incomplete_rbbb" not in _irbbb_codes(130, True)
    assert "incomplete_rbbb" not in _irbbb_codes(70, True)      # narrow rSr': normal variant
    assert "incomplete_rbbb" not in _irbbb_codes(105, False)
    # Children: an rSR' in V1 is a common normal variant -> never incomplete RBBB.
    m = Measurements(heart_rate_bpm=95, rhythm_regular=True, qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=130, qrs_ms=85, qt_ms=320, qtc_bazett_ms=400))
    kid = detect_findings(_sig(age_years=8), m, norms=load_norms("pediatric"), la_ra_reversed=False)
    assert "incomplete_rbbb" not in kid.codes()


def test_long_qt_is_qrs_adjusted():
    # QTc 470 with a 140-ms QRS: 40 ms of it is depolarization -> 430, not long QT.
    wide = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=False,
                        intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=440, qtc_bazett_ms=470))
    assert "long_qt" not in _find(wide).codes()
    # Same QTc with a narrow QRS stays long QT; a clearly long QTc stays long despite a wide QRS.
    narrow = Measurements(heart_rate_bpm=70, rhythm_regular=True,
                          intervals=Intervals(pr_ms=160, qrs_ms=90, qt_ms=440, qtc_bazett_ms=470))
    assert "long_qt" in _find(narrow).codes()
    very = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=False,
                        intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=500, qtc_bazett_ms=530))
    assert "long_qt" in _find(very).codes()
