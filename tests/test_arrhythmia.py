"""Beat-classifier / ventricular-ectopy tests (TZ §7.1).

CI-safe: no external MIT-BIH data. Validates the runtime contract (feature
extraction, no-op without a model, the committed model loads and behaves) and
the finding thresholds. Classifier accuracy is validated offline on MIT-BIH
(DS1->DS2, VEB ~0.82 sens / 0.97 PPV; see DECISIONS.md).
"""

import numpy as np

from cardiolens.arrhythmia import (
    BeatClassifier,
    EctopySummary,
    ectopy_findings,
    extract_beat_features,
    summarize_ectopy,
)
from cardiolens.arrhythmia.beats import FEATURES
from cardiolens.config import configs_dir
from cardiolens.layer1_ingestion import generate_synthetic_ecg


def test_ectopy_findings_thresholds():
    # <2 ventricular beats -> no finding (single beat within per-beat error).
    assert ectopy_findings(EctopySummary(n_beats=12, n_ventricular=1)) == []
    fs = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=4))
    assert [f.code for f in fs] == ["ventricular_ectopy"]
    big = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=6, bigeminy=True))
    assert [f.code for f in big] == ["ventricular_bigeminy"]
    # VT fires only on the gated vt flag (fast, >=4-beat run), critical, first.
    vt = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=5, max_vt_run=4, vt=True))
    assert [f.code for f in vt] == ["ventricular_tachycardia"]
    assert vt[0].critical is True
    # A bare run count without the vt gate must NOT raise a critical VT (a slow or
    # short V cluster is not VT) — guards against false §10 alerts.
    not_vt = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=3, max_vt_run=3, vt=False))
    assert "ventricular_tachycardia" not in [f.code for f in not_vt]
    # Exactly two consecutive V beats -> couplet.
    cpl = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=2, max_vt_run=2))
    assert [f.code for f in cpl] == ["ventricular_couplet"]


def test_unfitted_classifier_is_noop():
    clf = BeatClassifier.load(configs_dir() / "does_not_exist.json")
    assert clf.fitted is False
    assert clf.predict(np.zeros((3, len(FEATURES)))) == ["N", "N", "N"]


def test_committed_model_loads_and_predicts():
    clf = BeatClassifier.load(configs_dir() / "beat_lda.json")
    assert clf.fitted is True
    assert set(clf.classes) == {"N", "S", "V"}
    # Random feature rows classify to valid labels without error.
    rng = np.random.default_rng(0)
    preds = clf.predict(rng.normal(size=(5, len(clf.mu))))
    assert all(p in clf.classes for p in preds)


def test_no_false_ectopy_on_normal_synthetic():
    # A clean sinus rhythm must not raise ventricular ectopy.
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    import neurokit2 as nk
    lead = sig.lead("II")
    _, info = nk.ecg_peaks(lead, sampling_rate=sig.sampling_rate_hz)
    rpeaks = info["ECG_R_Peaks"]
    clf = BeatClassifier.load(configs_dir() / "beat_lda.json")
    summary = summarize_ectopy(lead, rpeaks, sig.sampling_rate_hz, clf)
    assert summary.n_beats > 0
    assert summary.burden <= 0.15  # essentially no ventricular beats on clean sinus


def test_feature_matrix_shape():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    import neurokit2 as nk
    lead = sig.lead("II")
    _, info = nk.ecg_peaks(lead, sampling_rate=sig.sampling_rate_hz)
    X, idx = extract_beat_features(lead, info["ECG_R_Peaks"], sig.sampling_rate_hz)
    assert X.shape[0] == len(idx)
    assert X.shape[1] == len(FEATURES)


def _pulse_train(rr_ms, odd=None, fs=500.0):
    """3-lead (II, V1, V5) train of narrow QRS-like pulses at the given RR
    intervals; beats listed in ``odd`` get a wide biphasic shape instead."""
    from cardiolens.contracts.signal import SignalRepresentation

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


def test_pvc_rule_finds_premature_differently_shaped_beat():
    from cardiolens.arrhythmia import detect_pvc_beats

    rr = [800] * 5 + [560, 1040] + [800] * 5          # beat 6: premature + compensatory
    sig, rp = _pulse_train(rr, odd={6})
    assert detect_pvc_beats(sig, rp, 500.0) == [6]
    sig, rp = _pulse_train([800] * 12)
    assert detect_pvc_beats(sig, rp, 500.0) == []
    # premature but same shape (e.g. a PAC) -> not a PVC
    sig, rp = _pulse_train(rr)
    assert detect_pvc_beats(sig, rp, 500.0) == []


def test_pvc_findings_bigeminy_pattern():
    from cardiolens.arrhythmia import pvc_findings

    assert [f.code for f in pvc_findings([2, 4, 6, 8], 12)] == ["ventricular_bigeminy"]
    assert [f.code for f in pvc_findings([5], 12)] == ["ventricular_ectopy"]
    assert pvc_findings([], 12) == []


def test_pvc_explains_af_like_irregularity():
    from cardiolens.pipeline import _pvc_explains_irregularity

    # Regular sinus with two PVC couplings: explained once the PVC intervals go.
    rr = [800] * 4 + [560, 1040] + [800] * 4 + [560, 1040] + [800] * 3
    rp = np.cumsum([0] + rr)
    assert _pvc_explains_irregularity(rp, [5, 11])[0] is True
    # Genuinely irregular background stays AF even after removing one PVC.
    rng = np.random.default_rng(1)
    rr_af = list(rng.uniform(450, 1100, 14))
    rp_af = np.cumsum([0] + rr_af)
    assert _pvc_explains_irregularity(rp_af, [6])[0] is False


def test_ventricular_run_detector():
    from cardiolens.arrhythmia import detect_ventricular_runs, vt_run_finding

    # Sinus 75/min, then three premature wide beats at 125/min, then sinus again.
    rr = [800] * 5 + [500, 480, 480, 1100] + [800] * 4
    sig, rp = _pulse_train(rr, odd={6, 7, 8})
    runs = detect_ventricular_runs(sig, rp, 500.0)
    assert runs == [(6, 3, 125)]
    f = vt_run_finding(runs)
    assert f.code == "ventricular_tachycardia" and f.critical
    # A couplet is not a run; a slow run (idioventricular, < 100/min) is not VT.
    sig, rp = _pulse_train([800] * 5 + [500, 480, 1100] + [800] * 5, odd={6, 7})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    sig, rp = _pulse_train([800] * 5 + [700, 700, 700, 1100] + [800] * 4, odd={6, 7, 8})
    assert detect_ventricular_runs(sig, rp, 500.0) == []
    # Same-shaped premature beats (atrial run) and plain sinus: no ventricular run.
    sig, rp = _pulse_train(rr)
    assert detect_ventricular_runs(sig, rp, 500.0) == []


def _atrial_strip(rr_ms, p_waves, fs=500.0, seed=0):
    """II/V1 strip: narrow QRS at the given RR; ``p_waves`` puts a P 160 ms before each
    QRS, otherwise low uncoordinated atrial activity fills the baseline."""
    from cardiolens.contracts.signal import SignalRepresentation

    rng = np.random.default_rng(seed)
    t_r = np.cumsum([600.0] + list(rr_ms)) / 1000.0
    n = int((t_r[-1] + 0.5) * fs)
    tt = np.arange(n) / fs
    x = np.zeros(n)
    for t in t_r:
        x += 1000 * np.exp(-((tt - t) / 0.012) ** 2)
        if p_waves:
            x += 150 * np.exp(-((tt - t + 0.16) / 0.025) ** 2)
    if not p_waves:
        phase = np.cumsum(2 * np.pi * rng.uniform(4.5, 7.5, n) / fs)
        x += 15 * np.sin(phase) * rng.uniform(0.5, 1.0, n)
    sig = SignalRepresentation(signal=np.stack([x, 0.8 * x]).astype(np.float32), sampling_rate_hz=fs,
                               lead_names=["II", "V1"], duration_s=n / fs, source="synthetic")
    return sig, (t_r * fs).astype(int)


def test_sinus_label_requires_organised_p_waves():
    from cardiolens.contracts.features import Intervals, Measurements
    from cardiolens.contracts.findings import Finding, FindingSet
    from cardiolens.pipeline import _check_sinus_p_waves

    def run(p_waves):
        sig, rp = _atrial_strip([900] * 10, p_waves)
        fs = FindingSet(findings=[Finding(code="sinus_rhythm", group="rhythm", label_ru="s",
                                          label_uz="s", confidence=0.9)])
        m = Measurements(heart_rate_bpm=67, r_peaks_samples=[int(r) for r in rp],
                         intervals=Intervals(qt_ms=380))
        _check_sinus_p_waves(fs, sig, m, [])
        return fs.codes()

    assert run(True) == {"sinus_rhythm"}
    assert run(False) == {"rhythm_without_p_waves"}
