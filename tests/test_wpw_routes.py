"""WPW routes after D65: the classic conjunction needs a slurred upstroke, and a
definite-LBBB width with a PR that is not short is a block, not a delta wave."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=50, sex="male"),
    )


def _codes(pr, qrs, delta, v1pos=False):
    m = Measurements(heart_rate_bpm=70, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     qrs_v1_terminal_positive=v1pos, qrs_v6_terminal_positive=not v1pos,
                     intervals=Intervals(pr_ms=pr, qrs_ms=qrs, qt_ms=380, qtc_bazett_ms=400),
                     morphology=Morphology(delta_ms=delta))
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_classic_route_needs_a_slurred_upstroke():
    assert "wpw_preexcitation" in _codes(pr=105, qrs=150, delta=60)       # short PR, wide, slurred
    assert "wpw_preexcitation" not in _codes(pr=105, qrs=150, delta=30)   # a block with a short PR
    assert "wpw_preexcitation" in _codes(pr=105, qrs=150, delta=None)     # no descriptor: the conjunction stands


def test_definite_lbbb_width_with_a_normal_pr_is_not_a_delta_wave():
    assert "wpw_preexcitation" not in _codes(pr=126, qrs=168, delta=66)   # MIMIC false pattern
    assert "wpw_preexcitation" in _codes(pr=126, qrs=112, delta=60)       # the delta route proper
    assert "wpw_preexcitation" in _codes(pr=110, qrs=168, delta=66)       # short PR: pre-excitation


def test_rbbb_morphology_still_blocks_the_delta_route():
    assert "wpw_preexcitation" not in _codes(pr=130, qrs=125, delta=55, v1pos=True)
