"""Serial comparison tests (TZ §8.4)."""

from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.findings import Finding, FindingSet
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.pipeline import run_pipeline
from cardiolens.serial import compare


def _fs(*codes):
    return FindingSet(findings=[
        Finding(code=c, label_ru=c, label_uz=c, confidence=0.8) for c in codes
    ])


def test_new_st_elevation_is_critical_delta():
    cur = compare(
        _fs("sinus_rhythm", "st_elevation"),
        Measurements(intervals=Intervals(qtc_bazett_ms=420)),
        _fs("sinus_rhythm"),
        Measurements(intervals=Intervals(qtc_bazett_ms=420)),
    )
    st = [d for d in cur.deltas if d.code == "st_elevation"]
    assert st and st[0].direction == "new" and st[0].critical


def test_resolved_finding_detected():
    c = compare(
        _fs("sinus_rhythm"),
        Measurements(),
        _fs("sinus_rhythm", "atrial_fibrillation"),
        Measurements(),
    )
    assert any(d.code == "atrial_fibrillation" and d.direction == "resolved"
               for d in c.deltas)


def test_qtc_prolongation_delta():
    c = compare(
        _fs("sinus_rhythm"),
        Measurements(intervals=Intervals(qtc_bazett_ms=520)),
        _fs("sinus_rhythm"),
        Measurements(intervals=Intervals(qtc_bazett_ms=430)),
    )
    assert c.qtc_delta_ms == 90.0
    assert any(d.code == "qtc_prolongation" for d in c.deltas)


def test_no_prior_no_serial():
    result = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    assert result.serial is None


def test_pipeline_serial_new_stemi_escalates():
    prior = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A")
    cur = generate_synthetic_ecg(heart_rate_bpm=70, st_elevation_uv=250,
                                 st_leads=["II", "III", "aVF"])
    result = run_pipeline(cur, mode="B", prior=prior)
    assert result.serial is not None
    assert any(d.code == "st_elevation" and d.critical for d in result.serial.deltas)
    # New critical dynamics force escalation and appear as a serial alert.
    assert result.report.mode == "escalate"
    assert any(a.code == "serial_st_elevation" for a in result.safety.alerts)
