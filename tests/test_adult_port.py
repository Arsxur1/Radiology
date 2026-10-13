"""Adult rhythm findings ported from claude/vibrant-curie-ro1zsz (D25, D37, D38, D40, D41):
ventricular runs (VT, §10), 12-lead PVCs / bigeminy, pauses, the signal-organization gate,
and the SVT / wide-complex-tachycardia split at 140/min."""

import numpy as np

from cardiolens.arrhythmia import (
    detect_pvc_beats,
    detect_ventricular_runs,
    pvc_findings,
    vt_run_finding,
)
from cardiolens.config import load_norms
from cardiolens.contracts.features import Intervals, Measurements
from cardiolens.contracts.findings import FindingSet
from cardiolens.contracts.signal import STANDARD_12_LEADS, Patient, SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings import detect_findings
from cardiolens.pipeline import run_pipeline
from cardiolens.safety import scan_critical


def _pulse_train(rr_ms, odd=None, fs=500.0):
    """3-lead (II, V1, V5) train of narrow QRS-like pulses at the given RR intervals;
    beats listed in ``odd`` get a wide biphasic shape instead."""
    t_r = np.cumsum([400.0] + list(rr_ms)) / 1000.0
    n = int((t_r[-1] + 0.5) * fs)
    x = np.zeros((3, n))
    tt = np.arange(n) / fs
    for k, t in enumerate(t_r):
        if odd and k in odd:
            pulse = 900 * (np.exp(-((tt - t + 0.03) / 0.03) ** 2) - np.exp(-((tt - t - 0.04) / 0.03) ** 2))
        else:
            pulse = 1000 * np.exp(-((tt - t) / 0.012) ** 2)
        x += pulse
    sig = SignalRepresentation(signal=x.astype(np.float32), sampling_rate_hz=fs,
                               lead_names=["II", "V1", "V5"], duration_s=n / fs, source="synthetic")
    return sig, (t_r * fs).astype(int)


def test_ventricular_run_detector():
    rr = [800] * 5 + [500, 480, 480, 1100] + [800] * 4
    sig, rp = _pulse_train(rr, odd={6, 7, 8})
    runs = detect_ventricular_runs(sig, rp, 500.0)
    assert runs == [(6, 3, 125)]
    f = vt_run_finding(runs)
    assert f.code == "ventricular_tachycardia" and f.critical
    assert any(a.code == "ventricular_tachycardia"
               for a in scan_critical(FindingSet(findings=[f]), Measurements()).alerts)
    # A couplet is not a run; a slow (idioventricular) run is not VT; same-shaped beats are not V.
    sig, rp = _pulse_train([800] * 5 + [500, 480, 1100] + [800] * 5, odd={6, 7})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    sig, rp = _pulse_train([800] * 5 + [700, 700, 700, 1100] + [800] * 4, odd={6, 7, 8})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    sig, rp = _pulse_train(rr)
    assert detect_ventricular_runs(sig, rp, 500.0) == []


def test_pvc_rule_and_bigeminy():
    rr = [800] * 5 + [560, 1040] + [800] * 5
    sig, rp = _pulse_train(rr, odd={6})
    assert detect_pvc_beats(sig, rp, 500.0) == [6]
    sig, rp = _pulse_train(rr)                     # premature but same shape (PAC)
    assert detect_pvc_beats(sig, rp, 500.0) == []
    assert [f.code for f in pvc_findings([2, 4, 6, 8], 12)] == ["ventricular_bigeminy"]
    assert [f.code for f in pvc_findings([5], 12)] == ["ventricular_ectopy"]


def test_pause_measured_flagged_and_critical():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    fs = int(sig.sampling_rate_hz)
    a = sig.signal.shape[1] // 2
    sig.signal[:, a:a + int(1.6 * fs)] = 0.0
    res = run_pipeline(sig, mode="A")
    assert res.measurements.max_rr_ms >= 1500 and res.measurements.pause_ratio >= 1.8
    assert "sinus_pause" in res.findings.codes()
    crit = scan_critical(FindingSet(), Measurements(max_rr_ms=3200))
    assert any(al.code == "ventricular_pause" for al in crit.alerts)


def test_disorganized_signal_withholds_findings():
    rng = np.random.RandomState(0)
    noise = SignalRepresentation(
        signal=rng.normal(0, 300, (12, 5000)).astype("float32"), sampling_rate_hz=500,
        lead_names=list(STANDARD_12_LEADS), duration_s=10, source="synthetic",
        patient=Patient(age_years=50, sex="male"))
    codes = run_pipeline(noise, mode="A").findings.codes()
    assert "poor_quality_signal" in codes
    assert not ({"sinus_bradycardia", "sinus_rhythm", "sinus_tachycardia"} & codes)
    clean = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A").findings.codes()
    assert "poor_quality_signal" not in clean


def _find(m):
    sig = SignalRepresentation(signal=np.zeros((12, 10)), sampling_rate_hz=500,
                               lead_names=list(STANDARD_12_LEADS), duration_s=0.02,
                               source="synthetic", patient=Patient(age_years=50, sex="male"))
    return detect_findings(sig, m, norms=load_norms("adult"))


def test_wide_tachycardia_is_not_sinus_and_narrow_is_svt():
    m = Measurements(heart_rate_bpm=175, rhythm_regular=True, qrs_v1_terminal_positive=False,
                     intervals=Intervals(pr_ms=100, qrs_ms=140, qt_ms=300, qtc_bazett_ms=510))
    fs = _find(m)
    assert "wide_complex_tachycardia" in fs.codes()
    assert not {"sinus_tachycardia", "wpw_preexcitation", "first_degree_av_block"} & fs.codes()
    assert [a.code for a in scan_critical(fs, m).alerts].count("wide_complex_tachycardia") == 1
    narrow = Measurements(heart_rate_bpm=175, rhythm_regular=True,
                          intervals=Intervals(pr_ms=120, qrs_ms=90, qt_ms=280, qtc_bazett_ms=470))
    assert "supraventricular_tachycardia" in _find(narrow).codes()
    slow = Measurements(heart_rate_bpm=120, rhythm_regular=True,
                        intervals=Intervals(pr_ms=150, qrs_ms=90, qt_ms=320, qtc_bazett_ms=450))
    assert "sinus_tachycardia" in _find(slow).codes()
    # Backstop alert from 140/min when Layer 4 did not emit the finding.
    assert any(a.code == "wide_complex_tachycardia" for a in
               scan_critical(FindingSet(), Measurements(heart_rate_bpm=145,
                                                        intervals=Intervals(qrs_ms=150))).alerts)


# --- Paced rhythm (source D24, D74, D75) ---
from cardiolens.layer3_measure.pacing import (  # noqa: E402
    pacing_multilead_spike_times,
    pacing_spike_coupling,
    pacing_spike_ratio,
)


def _r_times(sig):
    return [r / sig.sampling_rate_hz for r in
            run_pipeline(sig.model_copy(deep=True), mode="A").measurements.r_peaks_samples]


def _spiked_ecg(offset_s, leads=None, scale=3.0):
    base = generate_synthetic_ecg(heart_rate_bpm=72)
    r_s = _r_times(base)
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    fs = sig.sampling_rate_hz
    for name in (leads or ["II"]):
        li = sig.lead_names.index(name)
        amp = float(np.max(np.abs(sig.signal[li]))) * scale
        for t in r_s:
            i = int(round((t + offset_s) * fs))
            if 0 <= i < sig.signal.shape[1]:
                sig.signal[li, i] += amp
    return sig


def test_pacing_spike_coupling_math():
    r = [1.0, 2.0, 3.0, 4.0]
    assert pacing_spike_coupling([0.95, 1.95, 2.95, 3.95], r) == 1.0
    assert pacing_spike_coupling([1.5, 2.5], r) == 0.0
    assert pacing_spike_coupling([0.99], r) == 0.25


def test_paced_call_requires_spike_locked_to_qrs():
    assert pacing_spike_ratio(generate_synthetic_ecg(heart_rate_bpm=72)) < 5.5
    locked = run_pipeline(_spiked_ecg(-0.04), mode="A").findings.codes()
    assert "paced_rhythm" in locked and "sinus_rhythm" not in locked
    assert "paced_rhythm" not in run_pipeline(_spiked_ecg(+0.40), mode="A").findings.codes()


def test_multilead_spike_detects_small_qrs_locked_spikes():
    sig = _spiked_ecg(-0.04, leads=list(STANDARD_12_LEADS), scale=0.6)
    assert len(pacing_multilead_spike_times(sig)) >= 5
    one = _spiked_ecg(-0.04, leads=["II"], scale=0.6)
    assert pacing_multilead_spike_times(one) == []
