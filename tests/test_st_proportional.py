"""Discordant ST elevation out of proportion to the S wave is primary (D54).

Under LBBB / LVH the expected (secondary) repolarisation is discordant to the
QRS, so discordant elevation is silent for the STEMI-grade alert — unless it is
at least proportional_discordant_ratio of the lead's own S depth (Smith-modified
Sgarbossa / Armstrong): then it is a STEMI on top of the block.
"""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement, Voltages
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _lbbb_with_inferior_discordant_elevation(j_uv, s_depth_uv):
    st = [STMeasurement(lead=ld, j_point_uv=j_uv, j60_uv=j_uv, j80_uv=j_uv, qrs_net_uv=-s_depth_uv,
                        s_depth_uv=s_depth_uv, r_height_uv=100.0) for ld in ("II", "III", "aVF")]
    st += [STMeasurement(lead=ld, j_point_uv=0.0, j60_uv=0.0, j80_uv=0.0, qrs_net_uv=800.0, s_depth_uv=100.0, r_height_uv=900.0)
           for ld in ("I", "aVL", "V1", "V2", "V3", "V4", "V5", "V6")]
    return Measurements(heart_rate_bpm=75, rhythm_regular=True, rr_cv=0.03, p_wave_ratio=1.0,
                        qrs_v1_terminal_positive=False, voltages=Voltages(intrinsicoid_v5_v6_ms=70),
                        intervals=Intervals(pr_ms=160, qrs_ms=150, qt_ms=400, qtc_bazett_ms=430),
                        morphology=Morphology(st=st))


def _codes(m):
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False)


def test_proportional_discordant_elevation_under_lbbb_raises_the_alert():
    fs = _codes(_lbbb_with_inferior_discordant_elevation(j_uv=150.0, s_depth_uv=400.0))   # 0.375 of S
    assert "lbbb" in fs.codes() and "st_elevation" in fs.codes()
    alert = next(f for f in fs.findings if f.code == "st_elevation")
    assert any(e.feature == "discordant_proportional" and "II" in e.value for e in alert.evidence)


def test_expected_discordant_elevation_under_lbbb_stays_silent():
    fs = _codes(_lbbb_with_inferior_discordant_elevation(j_uv=150.0, s_depth_uv=1200.0))  # 0.125 of S
    assert "lbbb" in fs.codes() and "st_elevation" not in fs.codes()
    assert "secondary_st_t_change" in fs.codes()


def test_proportional_rule_needs_the_absolute_minimum():
    fs = _codes(_lbbb_with_inferior_discordant_elevation(j_uv=60.0, s_depth_uv=100.0))    # ratio 0.6 but 60 µV
    assert "st_elevation" not in fs.codes()
