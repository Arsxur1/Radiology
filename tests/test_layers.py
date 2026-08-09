"""Per-layer tests (TZ §16)."""

import pytest

from cardiolens.config import load_rule_base, norms_for_patient
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer2_preprocess import assess_quality, detect_la_ra_reversal, preprocess
from cardiolens.layer3_measure import measure
from cardiolens.layer3_measure.intervals import qtc_bazett, qtc_fridericia
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


# --- Layer 4: findings ---
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
