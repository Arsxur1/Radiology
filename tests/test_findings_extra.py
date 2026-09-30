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
