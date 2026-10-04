"""ST depression (D67): discordant depression under a block / LVH / WPW is the
secondary ST-T change, not the ischaemia code; concordant depression still is."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements, Morphology, STMeasurement
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _codes(st, qrs_ms=90, v6pos=False):
    m = Measurements(heart_rate_bpm=75, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     qrs_v1_terminal_positive=False, qrs_v6_terminal_positive=v6pos,
                     intervals=Intervals(pr_ms=150, qrs_ms=qrs_ms, qt_ms=380, qtc_bazett_ms=400),
                     morphology=Morphology(st=[STMeasurement(lead=lead, j_point_uv=j, j60_uv=j60, j80_uv=j60, qrs_net_uv=net)
                                               for lead, j, j60, net in st]))
    return detect_findings(_sig(), m, norms=load_norms("adult"), la_ra_reversed=False).codes()


def test_discordant_depression_under_lbbb_is_secondary_not_the_ischaemia_code():
    codes = _codes([("V5", -80, -180, 1200), ("V6", -90, -200, 1100), ("I", -60, -150, 900)], qrs_ms=150, v6pos=True)
    assert "lbbb" in codes
    assert "st_depression" not in codes and "secondary_st_t_change" in codes


def test_concordant_depression_under_lbbb_stays_primary():
    codes = _codes([("V1", -80, -180, -1500), ("V2", -90, -200, -1300)], qrs_ms=150, v6pos=True)
    assert "lbbb" in codes and "st_depression" in codes


def test_depression_without_a_block_is_the_ischaemia_code():
    codes = _codes([("V5", -80, -180, 1200), ("V6", -90, -200, 1100)])
    assert "st_depression" in codes and "secondary_st_t_change" not in codes
