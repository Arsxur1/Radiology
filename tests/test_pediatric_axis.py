"""Pediatric QRS-axis normal range by age band (D53).

A rightward axis is physiological in infants; one range to 180° for every child
scored 0 of 35 cohort children aged 4–15 labelled right axis deviation
(measured 88–134°). The upper limit now follows the age band.
"""

import numpy as np

from cardiolens.config import load_norms, norms_for_patient
from cardiolens.contracts.features import Axes, Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer4_findings.rules import _axis_range


def _sig(age_years=None, age_days=None):
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=0.02, source="synthetic",
        patient=Patient(age_years=age_years, age_days=age_days, sex="male"),
    )


def _m(axis):
    return Measurements(heart_rate_bpm=90, rhythm_regular=True, rr_cv=0.03, p_wave_ratio=1.0,
                        intervals=Intervals(pr_ms=130, qrs_ms=80, qt_ms=340, qtc_bazett_ms=400),
                        axes=Axes(qrs_deg=axis))


def _codes(age_years, axis, age_days=None):
    norms = norms_for_patient(age_years, age_days)
    return detect_findings(_sig(age_years, age_days), _m(axis), norms=norms, la_ra_reversed=False).codes()


def test_axis_upper_limit_falls_with_age():
    ped = load_norms("pediatric")
    hi = [_axis_range(ped, Patient(age_days=10))[1], _axis_range(ped, Patient(age_years=0.5))[1],
          _axis_range(ped, Patient(age_years=2))[1], _axis_range(ped, Patient(age_years=6))[1],
          _axis_range(ped, Patient(age_years=12))[1]]
    assert hi == sorted(hi, reverse=True) and hi[0] > hi[-1]
    assert hi[-1] < 180                                   # the old single range


def test_school_age_right_axis_is_called_and_infant_is_not():
    assert "right_axis_deviation" in _codes(10.0, 130.0)
    assert "right_axis_deviation" not in _codes(10.0, 85.0)
    assert "right_axis_deviation" not in _codes(0.5, 115.0)   # physiological in an infant
    assert "right_axis_deviation" not in _codes(None, 115.0, age_days=5)


def test_no_age_falls_back_to_the_table_range():
    ped = load_norms("pediatric")
    assert _axis_range(ped, Patient()) == tuple(ped["axis_deg"]["qrs_normal"])


def test_adult_axis_unchanged():
    assert "right_axis_deviation" in _codes(40.0, 95.0)
    assert "right_axis_deviation" not in _codes(40.0, 85.0)
