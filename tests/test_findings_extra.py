"""Tests for expanded §7 findings: axis, low voltage, AV block III, flutter."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Axes, Intervals, Measurements
from cardiolens.contracts.findings import FindingSet
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.pipeline import run_pipeline
from cardiolens.safety import scan_critical


def _sig(age_years=40):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=0.02, source="synthetic",
        patient=Patient(age_years=age_years, sex="male"),
    )


def _codes(m, age=40):
    return detect_findings(_sig(age), m, norms=load_norms("adult"),
                           la_ra_reversed=False).codes()


def test_left_axis_deviation():
    assert "left_axis_deviation" in _codes(
        Measurements(heart_rate_bpm=70, axes=Axes(qrs_deg=-60),
                     intervals=Intervals(qrs_ms=90)))


def test_right_axis_deviation():
    assert "right_axis_deviation" in _codes(
        Measurements(heart_rate_bpm=70, axes=Axes(qrs_deg=120),
                     intervals=Intervals(qrs_ms=90)))


def test_extreme_axis():
    assert "extreme_axis" in _codes(
        Measurements(heart_rate_bpm=70, axes=Axes(qrs_deg=-120),
                     intervals=Intervals(qrs_ms=90)))


def test_normal_axis_no_deviation():
    codes = _codes(Measurements(heart_rate_bpm=70, axes=Axes(qrs_deg=45),
                                intervals=Intervals(qrs_ms=90)))
    assert not ({"left_axis_deviation", "right_axis_deviation", "extreme_axis"} & codes)


def test_complete_av_block_is_finding_and_critical():
    m = Measurements(heart_rate_bpm=38, av_dissociation=True, atrial_rate_bpm=80,
                     intervals=Intervals(qrs_ms=90))
    fnd = detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)
    assert "complete_av_block" in fnd.codes()
    saf = scan_critical(fnd, m)
    assert any(a.code == "complete_av_block" for a in saf.alerts)


def test_atrial_flutter():
    assert "atrial_flutter" in _codes(
        Measurements(heart_rate_bpm=75, atrial_rate_bpm=300,
                     intervals=Intervals(qrs_ms=90)))


def test_disorganized_signal_withholds_findings():
    # Pure band-limited noise must not yield a confident bradycardia read off
    # spurious R-peaks — the §5 organization gate withholds and flags it instead.
    rng = np.random.RandomState(0)
    noise = SignalRepresentation(
        signal=rng.normal(0, 300, (12, 5000)).astype("float32"), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=10, source="synthetic",
        patient=Patient(age_years=50, sex="male"))
    codes = run_pipeline(noise, mode="A").findings.codes()
    assert "poor_quality_signal" in codes
    assert not ({"sinus_bradycardia", "sinus_rhythm", "sinus_tachycardia"} & codes)
    # A clean signal is never gated.
    clean = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.codes()
    assert "poor_quality_signal" not in clean


def test_wide_complex_tachycardia_critical_from_140():
    # A wide-complex regular tachy at 145 (the SVT-narrow band) must raise the
    # §10 "exclude VT" alert, not fall through — thresholds aligned at 140.
    m = Measurements(heart_rate_bpm=145, intervals=Intervals(qrs_ms=150))
    saf = scan_critical(FindingSet(), m)
    assert any(a.code == "wide_complex_tachycardia" for a in saf.alerts)
    # A narrow tachy at the same rate does NOT (it is SVT, handled in Layer 4).
    saf2 = scan_critical(FindingSet(), Measurements(heart_rate_bpm=145,
                                                    intervals=Intervals(qrs_ms=90)))
    assert not any(a.code == "wide_complex_tachycardia" for a in saf2.alerts)


def test_low_qrs_voltage_end_to_end():
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72, amplitude_scale=0.4),
                          mode="A")
    assert "low_qrs_voltage" in result.findings.codes()


def test_normal_amplitude_no_low_voltage():
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    assert "low_qrs_voltage" not in result.findings.codes()


def test_long_qt_is_a_finding_above_the_limit_but_critical_only_from_500():
    # D44: QTc above the sex limit marks ~20% of resting ECGs (GE 12SL agrees),
    # so the critical alert is reserved for the high-risk tier.
    for qtc, critical in ((470, False), (505, True)):
        m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90, qt_ms=qtc - 20,
                                                                  qtc_bazett_ms=qtc))
        fnd = detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)
        assert "long_qt" in fnd.codes()
        alerts = [a.code for a in scan_critical(fnd, m).alerts]
        assert ("long_qt" in alerts) is critical
    # The threshold is configurable.
    m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90, qt_ms=470, qtc_bazett_ms=490))
    fnd = detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)
    assert "long_qt" not in [a.code for a in scan_critical(fnd, m).alerts]
    assert "long_qt" in [a.code for a in scan_critical(fnd, m, long_qt_critical_ms=480).alerts]


def test_stemi_alert_tier_uses_median_beat_j_point_criterion():
    # D48: the st_elevation finding is unchanged; the critical alert needs the UDMI
    # J-point criterion on the median beat (or no median beat at all).
    from cardiolens.contracts.features import Morphology
    from cardiolens.contracts.findings import Finding
    from cardiolens.safety.critical import stemi_j_point_criterion

    flat = {ld: (20.0, 40.0) for ld in STANDARD_12_LEADS}
    inferior = dict(flat, II=(150.0, 180.0), III=(160.0, 190.0), aVF=(140.0, 170.0))
    v2v3 = dict(flat, V2=(180.0, 220.0), V3=(170.0, 210.0))
    assert stemi_j_point_criterion(inferior, "male", 60) is True
    assert stemi_j_point_criterion(flat, "male", 60) is False
    # V2-V3: 180 uV is below the 200 uV limit for men >= 40 but above 150 uV for women.
    assert stemi_j_point_criterion(v2v3, "male", 60) is False
    assert stemi_j_point_criterion(v2v3, "female", 60) is True
    assert stemi_j_point_criterion({}, "male", 60) is None

    def alerts(st):
        fnd = FindingSet(findings=[Finding(code="st_elevation", group="ischemia", label_ru="s",
                                           label_uz="s", confidence=0.8)])
        m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90),
                         morphology=Morphology(st_median_beat=st))
        return [a.code for a in scan_critical(fnd, m, patient=Patient(sex="male", age_years=60)).alerts]

    assert "st_elevation" in alerts(inferior)
    assert "st_elevation" not in alerts(flat)
    assert "st_elevation" in alerts({})          # no median beat -> conservative


def test_hyperkalaemia_alert_tier_tall_t_or_known_potassium():
    # D49: peaked_t_waves stays a finding; the critical alert needs a tall T on the
    # median beat (>= 1.0 mV precordial / 0.6 mV limb in >= 2 leads), a serum K+
    # >= 5.5 mmol/L, or no median beat at all (conservative).
    from cardiolens.contracts.features import Morphology
    from cardiolens.contracts.findings import Finding
    from cardiolens.contracts.signal import PatientContext
    from cardiolens.safety.critical import tall_t_criterion

    normal = {ld: 300.0 for ld in STANDARD_12_LEADS}
    tall = dict(normal, V2=1200.0, V3=1100.0)
    assert tall_t_criterion(tall) is True and tall_t_criterion(normal) is False
    assert tall_t_criterion(dict(normal, II=650.0, aVF=620.0)) is True    # limb >= 0.6 mV
    assert tall_t_criterion({}) is None

    def alerts(t, potassium=None):
        fnd = FindingSet(findings=[Finding(code="peaked_t_waves", group="other", label_ru="t",
                                           label_uz="t", confidence=0.7)])
        m = Measurements(heart_rate_bpm=70, intervals=Intervals(qrs_ms=90),
                         morphology=Morphology(t_median_beat=t))
        ctx = PatientContext(potassium=potassium)
        return [a.code for a in scan_critical(fnd, m, context=ctx).alerts]

    assert "peaked_t_waves" in alerts(tall)
    assert "peaked_t_waves" not in alerts(normal)
    assert "peaked_t_waves" in alerts(normal, potassium=6.1)
    assert "peaked_t_waves" in alerts({})


def test_severe_bradycardia_alert_uses_the_pals_limit_under_one_year():
    # D101 option (c): median rate < 60/min is critical under 1 year; < 40/min otherwise.
    from cardiolens.contracts.signal import Patient

    m = Measurements(heart_rate_bpm=55)
    infant = [a.code for a in scan_critical(FindingSet(), m, patient=Patient(age_years=0.3)).alerts]
    adult = [a.code for a in scan_critical(FindingSet(), m, patient=Patient(age_years=40)).alerts]
    unknown = [a.code for a in scan_critical(FindingSet(), m).alerts]
    assert "severe_bradycardia" in infant
    assert "severe_bradycardia" not in adult and "severe_bradycardia" not in unknown


def _ped_codes(m, **age):
    sig = _sig()
    sig.patient = Patient(sex="male", **age)
    return detect_findings(sig, m, norms=load_norms("pediatric"), la_ra_reversed=False).codes()


def test_newborn_axis_range_is_wider():
    # D114: axis 190 is normal for a newborn (p97.5 194) but right-axis deviation at 2 months.
    m = Measurements(heart_rate_bpm=140, axes=Axes(qrs_deg=190), intervals=Intervals(qrs_ms=60))
    assert "right_axis_deviation" not in _ped_codes(m, age_days=10)
    assert "right_axis_deviation" in _ped_codes(m, age_days=60)
    assert "right_axis_deviation" in _ped_codes(m, age_years=0)
    m.axes.qrs_deg = 200
    assert "right_axis_deviation" in _ped_codes(m, age_days=10)


def test_pediatric_peaked_t_limit_is_higher():
    # D118: T 500 uV, over half the QRS, in 2 leads: peaked for an adult, not for a child.
    from cardiolens.contracts.features import Morphology
    t = {"V2": 500.0, "V3": 500.0}
    q = {"V2": 800.0, "V3": 800.0}
    m = Measurements(heart_rate_bpm=90, intervals=Intervals(qrs_ms=80),
                     morphology=Morphology(t_median_beat=t, qrs_extreme_median_beat=q))
    assert "peaked_t_waves" in _codes(m)
    assert "peaked_t_waves" not in _ped_codes(m, age_years=8)
