"""Per-layer tests (TZ §16)."""

import numpy as np
import pytest

from cardiolens.config import load_app_config, load_rule_base, norms_for_patient
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer2_preprocess import assess_quality, detect_la_ra_reversal, preprocess
from cardiolens.layer3_measure import measure
from cardiolens.layer3_measure.intervals import qtc_bazett, qtc_fridericia
from cardiolens.layer3_measure.measure import _median_ms
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer5_causal import analyze
from cardiolens.safety import scan_critical


def _prep(sig):
    sig = preprocess(sig, mains_frequency_hz=50, bandpass_low_hz=0.5,
                     bandpass_high_hz=40, target_sampling_rate_hz=500)
    _, rev = assess_quality(sig, sqi_accept_threshold=0.5)
    return sig, rev


def _run(sig):
    sig, rev = _prep(sig)
    m = measure(sig)
    norms = norms_for_patient(sig.patient.age_years, sig.patient.age_days)
    fnd = detect_findings(sig, m, norms=norms, la_ra_reversed=rev)
    saf = scan_critical(fnd, m)
    return m, fnd, saf


# --- Layer 1/3: heart rate recovery ---
@pytest.mark.parametrize("hr", [50, 75, 110])
def test_heart_rate_recovered(hr):
    m, _, _ = _run(generate_synthetic_ecg(heart_rate_bpm=hr))
    assert m.heart_rate_bpm == pytest.approx(hr, abs=3)


def test_qrs_axis_and_intervals_plausible():
    m, _, _ = _run(generate_synthetic_ecg(heart_rate_bpm=75))
    assert 60 <= m.intervals.qrs_ms <= 110
    assert -30 <= m.axes.qrs_deg <= 90


def test_median_ms_rejects_implausible_beats():
    # Per-beat delineation outliers (a "PR" of ~70 ms, a negative QT) must be
    # dropped before the median so one bad fiducial can't drag the interval.
    fs = 500.0  # 1 ms = 0.5 samples
    good = [150 * fs / 1000, 152 * fs / 1000, 148 * fs / 1000]  # ~150 ms PRs
    noise = [70 * fs / 1000, -40 * fs / 1000]                   # impossible
    assert _median_ms(good + noise, fs, lo_ms=80, hi_ms=400) == pytest.approx(150)
    # Everything out of window -> no estimate rather than a fabricated one.
    assert _median_ms(noise, fs, lo_ms=80, hi_ms=400) is None
    # No bounds -> plain median (back-compat): 200 samples at 1 kHz = 200 ms.
    assert _median_ms([100, 200, 300], 1000.0) == pytest.approx(200)


def test_median_ms_preserves_extreme_pathology():
    # Upper windows must be supraphysiologic, not "normal": a genuine 720 ms long
    # QT, 220 ms wide QRS, 430 ms AV-block PR must survive so the safety layer can
    # fire — capping at normal limits would null the interval.
    fs = 1000.0
    assert _median_ms([720, 725, 715], fs, lo_ms=200, hi_ms=900) == pytest.approx(720)
    assert _median_ms([220, 222, 218], fs, lo_ms=40, hi_ms=300) == pytest.approx(220)
    assert _median_ms([430, 435, 425], fs, lo_ms=80, hi_ms=600) == pytest.approx(430)


def test_delineation_correction_shifts_pr():
    from cardiolens.config import DelineationCorrection

    sig, _ = _prep(generate_synthetic_ecg(heart_rate_bpm=70, pr_ms=160))
    base = measure(sig)
    # The per-lead fiducial PR takes the fiducial offset...
    fid = measure(sig, correction=DelineationCorrection(pr_ms=34.0))
    assert base.pr_ms_fiducial is not None and fid.pr_ms_fiducial is not None
    assert fid.pr_ms_fiducial == pytest.approx(base.pr_ms_fiducial + 34.0, abs=0.1)
    # ...and the reported global PR (D30) takes its own offset.
    glob = measure(sig, correction=DelineationCorrection(global_pr_ms=12.0))
    assert base.pr_global and glob.intervals.pr_ms == pytest.approx(base.intervals.pr_ms + 12.0, abs=0.1)


# --- QTc formulas ---
def test_qtc_formulas():
    # QT 400ms at RR 1s -> Bazett == 400, Fridericia == 400.
    assert qtc_bazett(400, 1.0) == pytest.approx(400)
    assert qtc_fridericia(400, 1.0) == pytest.approx(400)
    # Faster rate (RR 0.64) inflates Bazett QTc above raw QT.
    assert qtc_bazett(400, 0.64) > 400


# --- Layer 2: electrode reversal ---
def test_la_ra_reversal_detected():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    assert detect_la_ra_reversal(sig) is False
    i, a = sig.lead_names.index("I"), sig.lead_names.index("aVR")
    sig.signal[i] *= -1
    sig.signal[a] *= -1
    assert detect_la_ra_reversal(sig) is True


def test_pause_measured_and_flagged_critical():
    # Blank ~1.6 s of every lead in mid-record -> a dropped beat -> long RR gap.
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    fs = int(sig.sampling_rate_hz)
    a = sig.signal.shape[1] // 2
    sig.signal[:, a:a + int(1.6 * fs)] = 0.0
    m, _, _ = _run(sig)
    assert m.max_rr_ms is not None and m.max_rr_ms >= 1500
    assert m.pause_ratio is not None and m.pause_ratio >= 1.8
    # >=3 s would be critical; scan_critical flags it directly from the gap.
    from cardiolens.contracts.features import Measurements
    from cardiolens.safety import scan_critical
    crit = scan_critical(_run(generate_synthetic_ecg(heart_rate_bpm=70))[1],
                         Measurements(max_rr_ms=3200))
    assert any(al.code == "ventricular_pause" for al in crit.alerts)


def test_pacing_spike_ratio_separates_paced_from_clean():
    from cardiolens.layer3_measure.measure import pacing_spike_ratio
    # Clean sinus (no pacer) -> low slope-outlier ratio, below the 6.0 gate.
    clean = generate_synthetic_ecg(heart_rate_bpm=72)
    assert pacing_spike_ratio(clean) < 6.0
    # Inject narrow single-sample spikes (a pacer stimulus) into lead II -> the
    # max |diff| dwarfs the p99, so the ratio jumps well past the gate.
    paced = generate_synthetic_ecg(heart_rate_bpm=72)
    fs = int(paced.sampling_rate_hz)
    ii = paced.lead_names.index("II")
    amp = float(np.max(np.abs(paced.signal[ii]))) * 8.0
    for t in range(fs // 2, paced.signal.shape[1], fs):
        paced.signal[ii, t] += amp
    assert pacing_spike_ratio(paced) >= 6.0



def _spiked_ecg(offset_s):
    """Synthetic ECG with a pacer-like spike at (R peak + offset_s) on every beat."""
    from cardiolens.pipeline import run_pipeline
    base = generate_synthetic_ecg(heart_rate_bpm=72)
    r_s = [r / 500.0 for r in run_pipeline(base.model_copy(deep=True), mode="A")
           .measurements.r_peaks_samples]
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    fs = sig.sampling_rate_hz
    ii = sig.lead_names.index("II")
    amp = float(np.max(np.abs(sig.signal[ii]))) * 3.0   # one-sample step: extreme slope
    for t in r_s:
        i = int(round((t + offset_s) * fs))
        if 0 <= i < sig.signal.shape[1]:
            sig.signal[ii, i] += amp
    return sig


def test_paced_call_requires_spike_locked_to_qrs():
    from cardiolens.pipeline import run_pipeline
    locked = run_pipeline(_spiked_ecg(-0.04), mode="A")      # 40 ms before each R
    assert "paced_rhythm" in locked.findings.codes()
    # Same spikes mid-diastole (artifact-like, not QRS-locked) -> not paced.
    random_ = run_pipeline(_spiked_ecg(+0.40), mode="A")
    assert "paced_rhythm" not in random_.findings.codes()
    assert any("not QRS-locked" in a for a in random_.audit_log)


def test_pacing_spike_coupling_math():
    from cardiolens.layer3_measure.measure import pacing_spike_coupling
    r = [1.0, 2.0, 3.0, 4.0]
    assert pacing_spike_coupling([0.95, 1.95, 2.95, 3.95], r) == 1.0
    assert pacing_spike_coupling([1.5, 2.5], r) == 0.0
    assert pacing_spike_coupling([0.99], r) == 0.25


# --- Layer 4: findings ---
def test_svt_vs_sinus_tachycardia():
    # A regular narrow-complex tachy >=140/min is SVT, not sinus tachycardia.
    _, fast, _ = _run(generate_synthetic_ecg(heart_rate_bpm=155))
    assert "supraventricular_tachycardia" in fast.codes()
    assert "sinus_tachycardia" not in fast.codes()
    # A moderate tachy stays sinus tachycardia.
    _, mod, _ = _run(generate_synthetic_ecg(heart_rate_bpm=115))
    assert "sinus_tachycardia" in mod.codes()
    assert "supraventricular_tachycardia" not in mod.codes()


def test_bradycardia_finding():
    _, fnd, _ = _run(generate_synthetic_ecg(heart_rate_bpm=44))
    assert "sinus_bradycardia" in fnd.codes()


def test_av_block_finding():
    _, fnd, _ = _run(generate_synthetic_ecg(heart_rate_bpm=70, pr_ms=280))
    assert "first_degree_av_block" in fnd.codes()


def test_stemi_critical_alert():
    _, fnd, saf = _run(generate_synthetic_ecg(
        heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]))
    assert "st_elevation" in fnd.codes()
    assert saf.has_critical
    assert any(a.code == "st_elevation" for a in saf.alerts)


def test_pediatric_juvenile_t_not_pathologic():
    # Child with T inversion in V1-V4 should NOT be flagged as ischemia.
    sig = generate_synthetic_ecg(heart_rate_bpm=95, age_years=6)
    _, fnd, _ = _run(sig)
    assert "t_inversion" not in fnd.codes()


# --- Layer 5: causal engine ---
def test_hyperkalemia_chain_activates():
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=750, potassium=6.5)
    _, fnd, _ = _run(sig)
    g = analyze(fnd, load_rule_base(), sig.context)
    assert "hyperkalemia_severity_chain" in g.activated_rules
    assert "peaked_t_waves" in [n.id for n in g.nodes if n.observed]
    assert g.critical_flags  # chain is critical


def test_reversal_flagged_as_artifact_chain():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    i, a = sig.lead_names.index("I"), sig.lead_names.index("aVR")
    sig.signal[i] *= -1
    sig.signal[a] *= -1
    _, fnd, _ = _run(sig)
    g = analyze(fnd, load_rule_base(), sig.context)
    assert "electrode_reversal_la_ra" in g.activated_rules


def test_robust_rr_dispersion_low_for_regular_high_for_irregular():
    """rr_mad_cv: ~0 for a regular rhythm, large for a broadly irregular one."""
    reg, _ = _prep(generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10))
    irr, _ = _prep(generate_synthetic_ecg(heart_rate_bpm=90, duration_s=10, rr_jitter_ms=180))
    m_reg, m_irr = measure(reg), measure(irr)
    assert m_reg.rr_mad_cv is not None and m_reg.rr_mad_cv < 0.03
    assert m_irr.rr_mad_cv > 0.08


def test_preprocess_and_pipeline_leave_the_input_untouched():
    import numpy as np

    from cardiolens.layer1_ingestion import generate_synthetic_ecg
    from cardiolens.layer2_preprocess import preprocess
    from cardiolens.pipeline import run_pipeline

    sig = generate_synthetic_ecg(heart_rate_bpm=70, duration_s=10)
    before, fs = sig.signal.copy(), sig.sampling_rate_hz
    out = preprocess(sig, mains_frequency_hz=50, bandpass_low_hz=0.5, bandpass_high_hz=40,
                     target_sampling_rate_hz=250)
    assert out.sampling_rate_hz == 250 and sig.sampling_rate_hz == fs
    assert np.array_equal(sig.signal, before)
    first = run_pipeline(sig, mode="A")
    assert np.array_equal(sig.signal, before)
    assert run_pipeline(sig, mode="A").findings.codes() == first.findings.codes()


def test_wfdb_header_demographics_are_read(tmp_path):
    import numpy as np
    import wfdb

    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer1_ingestion.parsers import patient_from_comments

    assert patient_from_comments(["Age: 74", "Sex: Female", "Dx: 426783006"]).sex == "female"
    assert patient_from_comments(["<age>: 65 <sex>: M <diagnoses> CAD"]).age_years == 65.0
    p = patient_from_comments(["Age: NaN", "Sex: Unknown"])
    assert p.age_years is None and p.sex is None
    x = (np.sin(np.linspace(0, 20, 5000))[:, None] * np.ones((1, 2))).astype(np.float64)
    wfdb.wrsamp("rec", fs=500, units=["mV", "mV"], sig_name=["I", "II"], p_signal=x,
                comments=["Age: 8", "Sex: Male"], write_dir=str(tmp_path))
    sig = load_wfdb(tmp_path / "rec")
    assert sig.patient.age_years == 8.0 and sig.patient.sex == "male"


def _multilead_spiked_ecg(offset_s, leads="all", scale=0.6):
    """Synthetic ECG with a small one-sample spike on many leads at (R + offset_s)."""
    from cardiolens.pipeline import run_pipeline
    base = generate_synthetic_ecg(heart_rate_bpm=72)
    r_s = [r / 500.0 for r in run_pipeline(base.model_copy(deep=True), mode="A")
           .measurements.r_peaks_samples]
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    fs = sig.sampling_rate_hz
    rows = range(sig.signal.shape[0]) if leads == "all" else [sig.lead_names.index(leads)]
    for li in rows:
        d = np.abs(np.diff(sig.signal[li]))
        amp = float(np.percentile(d, 99)) * 3.0 * scale / 0.6
        for t in r_s:
            i = int(round((t + offset_s) * fs))
            if 0 <= i < sig.signal.shape[1]:
                sig.signal[li, i] += amp
    return sig


def test_multilead_spike_detects_small_qrs_locked_spikes():
    """D75: a modest spike on every lead at once, locked to the QRS -> paced; the same
    spike on one lead only is not a multi-lead spike."""
    from cardiolens.layer3_measure.measure import pacing_multilead_spike_times, pacing_spike_ratio
    from cardiolens.pipeline import run_pipeline
    sig = _multilead_spiked_ecg(-0.04)
    assert pacing_spike_ratio(sig) < 5.5                     # too small for the single-lead rule
    assert len(pacing_multilead_spike_times(sig)) >= 5
    assert "paced_rhythm" in run_pipeline(sig, mode="A").findings.codes()
    one = _multilead_spiked_ecg(-0.04, leads="II")
    assert pacing_multilead_spike_times(one) == []
    assert "paced_rhythm" not in run_pipeline(one, mode="A").findings.codes()
    late = _multilead_spiked_ecg(+0.40)                     # mid-diastole: not QRS-locked
    assert "paced_rhythm" not in run_pipeline(late, mode="A").findings.codes()


def test_lbbb_withholds_brugada_pattern():
    # D91: discordant ST elevation in V1-V3 under LBBB is not a Brugada pattern.
    from cardiolens.contracts.features import Measurements
    from cardiolens.contracts.findings import Finding, FindingSet
    from cardiolens.pipeline import _apply_rhythm_hierarchy

    def fs(*codes):
        return FindingSet(findings=[Finding(code=c, label_ru=c, label_uz=c, confidence=0.5)
                                    for c in codes])
    audit: list[str] = []
    f = fs("lbbb", "brugada_pattern")
    _apply_rhythm_hierarchy(f, Measurements(), audit)
    assert f.codes() == {"lbbb"} and audit
    f = fs("rbbb", "brugada_pattern")
    _apply_rhythm_hierarchy(f, Measurements(), [])
    assert "brugada_pattern" in f.codes()


def test_pediatric_af_with_recurring_p_becomes_sinus_arrhythmia(monkeypatch):
    # D98: children only — AF call with a recurring pre-QRS P in II and V1 is replaced.
    import cardiolens.pipeline as pl
    from cardiolens.contracts.features import Measurements
    from cardiolens.contracts.findings import Finding, FindingSet

    monkeypatch.setattr(pl, "pre_qrs_consistency", lambda *a, **k: (0.9, {"II": 0.95, "V1": 0.9}))
    monkeypatch.setattr(pl, "_flutter_hedge", lambda *a, **k: None)

    def run(pediatric):
        fs = FindingSet(findings=[Finding(code="atrial_fibrillation", label_ru="ФП", label_uz="AF",
                                          confidence=0.6)])
        pl._af_atrial_evidence(fs, None, Measurements(), [], pediatric=pediatric)
        return fs.codes()

    assert run(True) == {"sinus_arrhythmia"}
    assert run(False) == {"atrial_fibrillation"}      # adults keep the hedged AF call (D51)


def test_single_lead_pediatric_af_needs_only_that_lead(monkeypatch):
    # D101: a bedside single-lead strip measures P consistency in one lead only; in children
    # that is enough. A 12-lead child ECG still needs both II and V1, adults always do.
    from types import SimpleNamespace

    import cardiolens.pipeline as pl
    from cardiolens.contracts.features import Measurements
    from cardiolens.contracts.findings import Finding, FindingSet

    monkeypatch.setattr(pl, "pre_qrs_consistency", lambda *a, **k: (0.9, {"II": 0.9}))
    monkeypatch.setattr(pl, "_flutter_hedge", lambda *a, **k: None)

    def run(leads, pediatric):
        fs = FindingSet(findings=[Finding(code="atrial_fibrillation", label_ru="ФП", label_uz="AF",
                                          confidence=0.6)])
        pl._af_atrial_evidence(fs, SimpleNamespace(lead_names=leads), Measurements(), [],
                               pediatric=pediatric)
        return fs.codes()

    assert run(["II"], True) == {"sinus_arrhythmia"}
    assert run(["I", "II", "III", "V1"], True) == {"atrial_fibrillation"}
    assert run(["II"], False) == {"atrial_fibrillation"}


def test_infant_bradycardia_episode_in_a_normal_rate_strip():
    # D101: under 1 year the slowest instantaneous rate (longest RR) below the age norm is
    # named even when the median rate is normal; it keeps the strip out of autonomous mode B.
    from cardiolens.contracts.features import Measurements
    from cardiolens.contracts.findings import FindingSet
    from cardiolens.contracts.signal import Patient
    from cardiolens.layer4_findings.rules import _bradycardia_episode

    norms = norms_for_patient(0.05, 20)
    m = Measurements(heart_rate_bpm=140, max_rr_ms=800)          # slowest 75/min
    fs = FindingSet()
    assert _bradycardia_episode(m, norms, Patient(age_years=0.05, age_days=20), 90, fs)
    f = fs.by_code("bradycardia_episode")
    assert f.confidence < load_app_config().confidence.autonomous_min_component
    assert not _bradycardia_episode(Measurements(heart_rate_bpm=140, max_rr_ms=600), norms,
                                    Patient(age_years=0.05), 90, FindingSet())   # 100/min
    assert not _bradycardia_episode(m, norms_for_patient(5, None), Patient(age_years=5), 90,
                                    FindingSet())                                  # child
