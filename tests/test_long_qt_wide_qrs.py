"""Long QT under a wide QRS: the rule judges QT - 0.485 x QRS (Bogossian), D60."""

import numpy as np

from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer4_findings import detect_findings


def _sig():
    return SignalRepresentation(
        signal=np.zeros((12, 10)), sampling_rate_hz=500, lead_names=list(STANDARD_12_LEADS),
        duration_s=0.02, source="synthetic", patient=Patient(age_years=60, sex="male"),
    )


def _codes(qt_ms, qrs_ms, norms=None):
    # HR 60: Bazett QTc == QT (hand-built Measurements carry Bazett only)
    m = Measurements(heart_rate_bpm=60, rhythm_regular=True, rr_cv=0.02, p_wave_ratio=1.0,
                     intervals=Intervals(qrs_ms=qrs_ms, qt_ms=qt_ms, qtc_bazett_ms=qt_ms))
    return detect_findings(_sig(), m, norms=norms or load_norms("adult"), la_ra_reversed=False)


def test_wide_qrs_qt_is_judged_after_the_bogossian_correction():
    fs = _codes(qt_ms=480, qrs_ms=160)           # 480 - 0.485 x 160 = 402 ms: not long
    assert "long_qt" not in fs.codes()
    fs = _codes(qt_ms=480, qrs_ms=90)            # narrow QRS: 480 > 450, long
    assert "long_qt" in fs.codes()


def test_a_truly_long_qt_survives_the_correction_and_says_so():
    fs = _codes(qt_ms=560, qrs_ms=160)           # 560 - 77.6 = 482 > 450
    f = fs.by_code("long_qt")
    assert f is not None
    assert any(e.feature == "qrs_adjustment" and e.value == "bogossian" for e in f.evidence)
    assert "широкий QRS" in f.label_ru


def test_without_the_table_entries_the_rule_is_unchanged():
    norms = load_norms("adult")
    norms["qtc_ms"] = {k: v for k, v in norms["qtc_ms"].items() if not k.startswith("wide_qrs")}
    assert "long_qt" in _codes(qt_ms=480, qrs_ms=160, norms=norms).codes()
