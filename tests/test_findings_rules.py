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


def test_long_qt_uses_plain_qtc_even_with_wide_qrs():
    # D28: the QRS-adjusted limit was reverted (lost sensitivity on held-out
    # data); a critical long-QT call must not be discounted by QRS width.
    wide = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=False,
                        intervals=Intervals(pr_ms=160, qrs_ms=140, qt_ms=440, qtc_bazett_ms=470))
    assert "long_qt" in _find(wide).codes()


def _q_codes(q_waves, qrs_ms=95, v1_term=None, pr_ms=160, norms="adult", age_years=60):
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=v1_term,
                     intervals=Intervals(pr_ms=pr_ms, qrs_ms=qrs_ms, qt_ms=390, qtc_bazett_ms=420),
                     morphology=Morphology(q_waves=q_waves))
    return detect_findings(_sig(age_years=age_years), m, norms=load_norms(norms), la_ra_reversed=False)


_NO_Q = (0.0, 0.0, 800.0, False)


def test_pathological_q_inferior_contiguous_leads():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["III"] = (44.0, 300.0, 400.0, False)
    q["aVF"] = (40.0, 250.0, 500.0, False)
    fs = _q_codes(q)
    f = fs.by_code("pathological_q_waves")
    assert f is not None and "III,aVF" in f.label_ru and "нижняя" in f.label_ru


def test_pathological_q_needs_two_contiguous_leads_and_real_depth():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["III"] = (44.0, 300.0, 400.0, False)                 # isolated Q in III: normal variant
    assert "pathological_q_waves" not in _q_codes(q).codes()
    q["aVF"] = (40.0, 60.0, 900.0, False)                  # shallow Q (< 100 µV, < 25% of R)
    assert "pathological_q_waves" not in _q_codes(q).codes()
    q["aVL"] = (40.0, 300.0, 300.0, False)                 # I-aVL / III-aVL are not contiguous
    assert "pathological_q_waves" not in _q_codes(q).codes()


def test_pathological_q_v2_v3_lower_limit_and_v1_ignored():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["V1"] = (0.0, 0.0, 10.0, True)                       # QS in V1 alone: normal variant
    q["V2"] = (25.0, 80.0, 600.0, False)                   # any Q > 20 ms counts in V2-V3
    assert "pathological_q_waves" not in _q_codes(q).codes()
    q["V3"] = (0.0, 0.0, 20.0, True)
    assert "pathological_q_waves" in _q_codes(q).codes()


def test_pathological_q_suppressed_by_lbbb_and_absent_in_children():
    q = {ld: _NO_Q for ld in STANDARD_12_LEADS}
    q["V2"] = q["V3"] = (0.0, 0.0, 20.0, True)             # QS V1-V3 is expected in LBBB
    assert "pathological_q_waves" in _q_codes(q).codes()
    assert "pathological_q_waves" not in _q_codes(q, qrs_ms=150, v1_term=False).codes()
    assert "pathological_q_waves" not in _q_codes(q, norms="pediatric", age_years=8).codes()


def test_wide_tachycardia_is_not_labelled_sinus_or_preexcitation():
    from cardiolens.safety import scan_critical

    # INCART: a 175/min VT run read as "sinus tachycardia" + WPW (short "PR").
    m = Measurements(heart_rate_bpm=175, rhythm_regular=True, qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=100, qrs_ms=140, qt_ms=300, qtc_bazett_ms=510))
    fs = _find(m)
    assert "wide_complex_tachycardia" in fs.codes()
    assert not {"sinus_tachycardia", "wpw_preexcitation", "first_degree_av_block"} & fs.codes()
    alerts = [a.code for a in scan_critical(fs, m).alerts]
    assert alerts.count("wide_complex_tachycardia") == 1
    # Below the 140/min band a wide QRS tachycardia stays sinus tachycardia + BBB.
    slow = Measurements(heart_rate_bpm=120, rhythm_regular=True, qrs_v1_terminal_positive=True,
                        intervals=Intervals(pr_ms=150, qrs_ms=140, qt_ms=320, qtc_bazett_ms=450))
    codes = _find(slow).codes()
    assert "sinus_tachycardia" in codes and "wide_complex_tachycardia" not in codes
    # A narrow regular tachycardia at the same rate is still SVT.
    narrow = Measurements(heart_rate_bpm=175, rhythm_regular=True,
                          intervals=Intervals(pr_ms=120, qrs_ms=90, qt_ms=280, qtc_bazett_ms=470))
    assert "supraventricular_tachycardia" in _find(narrow).codes()


def test_short_pr_with_rbbb_morphology_stays_rbbb():
    """D58: a short global PR on a wide QRS with a terminal R in V1 is RBBB, not WPW."""
    rbbb = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=True,
                        intervals=Intervals(pr_ms=100, qrs_ms=140, qt_ms=420, qtc_bazett_ms=450))
    codes = _find(rbbb).codes()
    assert "rbbb" in codes and "wpw_preexcitation" not in codes
    # Without RBBB morphology the same short PR + wide QRS is still pre-excitation.
    wpw = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_v1_terminal_positive=False,
                       intervals=Intervals(pr_ms=100, qrs_ms=140, qt_ms=420, qtc_bazett_ms=450))
    assert "wpw_preexcitation" in _find(wpw).codes()


def test_wpw_from_short_pr_and_delta_wave():
    """D59: short PR + slurred QRS upstroke is pre-excitation even with a QRS < 110 ms;
    a short PR with a brisk upstroke is not."""
    def codes(ratio, qrs=104, pr=110):
        m = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_initial_energy_ratio=ratio,
                         intervals=Intervals(pr_ms=pr, qrs_ms=qrs, qt_ms=400, qtc_bazett_ms=430))
        return _find(m).codes()

    assert "wpw_preexcitation" in codes(0.15)
    assert "wpw_preexcitation" not in codes(0.40)
    assert "wpw_preexcitation" not in codes(0.15, pr=150)


def test_delta_wave_rule_leaves_rbbb_morphology_to_bbb_rule():
    """D60: short PR + slurred upstroke with a terminal R in V1 and QRS >= 130 ms stays RBBB."""
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_initial_energy_ratio=0.15,
                     qrs_v1_terminal_positive=True,
                     intervals=Intervals(pr_ms=110, qrs_ms=140, qt_ms=420, qtc_bazett_ms=450))
    codes = _find(m).codes()
    assert "rbbb" in codes and "wpw_preexcitation" not in codes


def test_low_voltage_needs_every_limb_or_every_precordial_lead_small():
    """D61: one large limb lead rules out limb low voltage; small precordials alone suffice."""
    def codes(limb, prec):
        m = Measurements(heart_rate_bpm=70, rhythm_regular=True, qrs_amplitude_limb_uv=300,
                         qrs_amplitude_limb_max_uv=limb, qrs_amplitude_precordial_max_uv=prec,
                         intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=400, qtc_bazett_ms=420))
        return _find(m).codes()

    assert "low_qrs_voltage" in codes(450, 1500)
    assert "low_qrs_voltage" in codes(900, 700)
    assert "low_qrs_voltage" not in codes(900, 1500)


def _t_measure(t):
    from cardiolens.contracts.features import Morphology
    return Measurements(heart_rate_bpm=70, rhythm_regular=True,
                        intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=400, qtc_bazett_ms=420),
                        morphology=Morphology(t_median_beat=t))


def test_t_inversion_needs_a_territory_on_the_median_beat():
    """D63: inverted T in aVR, V1 or III alone is not T inversion; two inferior leads are."""
    base = {"I": 200, "II": 250, "III": 50, "aVR": -220, "aVL": 80, "aVF": 150,
            "V1": -150, "V2": 300, "V3": 350, "V4": 320, "V5": 280, "V6": 220}
    assert "t_inversion" not in _find(_t_measure(base)).codes()
    assert "t_inversion" not in _find(_t_measure({**base, "III": -200})).codes()
    assert "t_inversion" in _find(_t_measure({**base, "III": -200, "aVF": -150})).codes()


def test_flat_t_counts_only_leads_where_t_is_upright():
    """D63: flat T in III/aVL/aVR/V1 is normal; three flat upright leads are flat T."""
    base = {"I": 200, "II": 250, "III": 20, "aVR": -220, "aVL": 30, "aVF": 150,
            "V1": 10, "V2": 300, "V3": 350, "V4": 320, "V5": 280, "V6": 220}
    assert "flat_t_waves" not in _find(_t_measure(base)).codes()
    assert "flat_t_waves" in _find(_t_measure({**base, "I": 40, "V5": 60, "V6": 50})).codes()


def test_st_findings_read_the_median_beat_when_available():
    """D64: per-lead J+60 from the median beat overrides the per-beat ST values."""
    from cardiolens.contracts.features import Morphology, STMeasurement
    per_beat = [STMeasurement(lead=l, j_point_uv=0, j60_uv=150) for l in ("II", "III", "aVF")]
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True,
                     intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=400, qtc_bazett_ms=420),
                     morphology=Morphology(st=per_beat))
    assert "st_elevation" in _find(m).codes()
    flat = {l: (0.0, 20.0) for l in ("I", "II", "III", "aVR", "aVL", "aVF",
                                     "V1", "V2", "V3", "V4", "V5", "V6")}
    m.morphology.st_median_beat = flat
    assert "st_elevation" not in _find(m).codes()


def test_peaked_t_reads_the_median_beat_against_the_largest_qrs_deflection():
    """D65: T > 400 µV and > 0.5 x largest QRS deflection in >= 2 leads on the median beat;
    the per-beat list is ignored when the median beat exists; tall T alone still fires."""
    from cardiolens.contracts.features import Morphology
    leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    t = {l: 300.0 for l in leads}
    q = {l: 1200.0 for l in leads}

    def codes(t, q, per_beat=()):
        m = Measurements(heart_rate_bpm=70, rhythm_regular=True,
                         intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=400, qtc_bazett_ms=420),
                         morphology=Morphology(t_median_beat=t, qrs_extreme_median_beat=q,
                                               peaked_t_leads=list(per_beat)))
        return _find(m).codes()

    assert "peaked_t_waves" not in codes(t, q, per_beat=("V2", "V3", "V4"))
    # 700 µV T beside a 1200 µV QRS: peaked in two leads.
    assert "peaked_t_waves" in codes({**t, "V2": 700, "V3": 700}, q)
    # Same T beside a 2000 µV QRS is proportionate, not peaked.
    assert "peaked_t_waves" not in codes({**t, "V2": 700, "V3": 700}, {**q, "V2": 2000, "V3": 2000})
    # Tall T (>= 1000 µV in two precordial leads) keeps the finding for the hyperK alert.
    assert "peaked_t_waves" in codes({**t, "V2": 1100, "V3": 1100}, {**q, "V2": 3000, "V3": 3000})
    # No median beat: per-beat fallback.
    assert "peaked_t_waves" in codes({}, {}, per_beat=("V2", "V3"))


def test_sokolow_lyon_from_median_beat_r_and_s():
    """D66: Sokolow-Lyon = |S V1| + max(R V5, R V6) from median-beat R/S; None without V1/V5/V6."""
    from cardiolens.layer3_measure.measure import _sokolow_lyon_median
    rs = {"V1": (200.0, -1500.0), "V5": (2100.0, -300.0), "V6": (1800.0, -100.0)}
    assert _sokolow_lyon_median(rs) == 3600.0
    assert _sokolow_lyon_median({k: v for k, v in rs.items() if k != "V6"}) is None
    assert _sokolow_lyon_median({}) is None


def test_qrs_axis_from_median_beat_net_deflection():
    """D69: axis = atan2(net aVF, net I) with net = R + S on the median beat."""
    from cardiolens.layer3_measure.measure import _qrs_axis_median
    assert _qrs_axis_median({"I": (800.0, -200.0), "aVF": (300.0, -300.0)}) == 0.0
    assert _qrs_axis_median({"I": (800.0, -100.0), "aVF": (100.0, -800.0)}) == -45.0
    assert _qrs_axis_median({"I": (100.0, -100.0), "aVF": (600.0, -100.0)}) == 90.0
    assert _qrs_axis_median({"I": (800.0, -100.0)}) is None


def test_lbbb_needs_full_width_and_deep_s_in_v1_v2():
    """D70: LBBB needs QRS >= 128 ms and, on the median beat, S <= -1000 µV in V1 or V2;
    otherwise the wide QRS is nonspecific."""
    from cardiolens.contracts.features import Morphology

    def codes(qrs, s_v1=None, s_v2=None):
        rs = {} if s_v1 is None else {"V1": (100.0, s_v1), "V2": (200.0, s_v2)}
        m = Measurements(heart_rate_bpm=75, rhythm_regular=True, qrs_v1_terminal_positive=False,
                         intervals=Intervals(pr_ms=160, qrs_ms=qrs, qt_ms=420, qtc_bazett_ms=440),
                         morphology=Morphology(qrs_rs_median_beat=rs))
        return _find(m).codes()

    assert "lbbb" in codes(150, -1500.0, -2500.0)
    assert "lbbb" in codes(150, -600.0, -1200.0)          # deep S in V2 is enough
    shallow = codes(150, -600.0, -700.0)                   # wide QRS without a QS/rS in V1-V2
    assert "lbbb" not in shallow and "wide_qrs_nonspecific" in shallow
    narrow = codes(124, -1500.0, -2500.0)                  # 120-127 ms: not a complete LBBB
    assert "lbbb" not in narrow and "wide_qrs_nonspecific" in narrow
    assert "lbbb" in codes(140)                            # no median beat: width only
