"""Layer 6 + grounding + mode-policy tests (TZ §9, §2, §16)."""

from cardiolens.config import load_app_config, load_rule_base, norms_for_patient
from cardiolens.contracts.report import Claim, InterpretationReport
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer2_preprocess import assess_quality, preprocess
from cardiolens.layer3_measure import measure
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer5_causal import analyze
from cardiolens.layer6_interpret import InterpretationInput, StubProvider, interpret
from cardiolens.layer6_interpret.grounding import validate_grounding
from cardiolens.safety import scan_critical


def _pipeline_to_input(sig, mode):
    cfg = load_app_config()
    sig = preprocess(sig, mains_frequency_hz=50, bandpass_low_hz=0.5,
                     bandpass_high_hz=40, target_sampling_rate_hz=500)
    _, rev = assess_quality(sig, sqi_accept_threshold=0.5)
    m = measure(sig)
    norms = norms_for_patient(sig.patient.age_years, sig.patient.age_days)
    fnd = detect_findings(sig, m, norms=norms, la_ra_reversed=rev)
    saf = scan_critical(fnd, m)
    g = analyze(fnd, load_rule_base(), sig.context)
    data = InterpretationInput(
        findings=fnd, causal_graph=g, measurements=m, patient=sig.patient,
        context=sig.context, mode=mode,
        critical_labels=[(a.label_ru, a.label_uz) for a in saf.alerts],
        disclaimer_ru=cfg.disclaimers.ru, disclaimer_uz=cfg.disclaimers.uz,
    )
    return data, saf, cfg


def test_stub_report_is_bilingual_and_grounded():
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.5)
    data, saf, cfg = _pipeline_to_input(sig, "A")
    rep = interpret(data, saf, provider=StubProvider(),
                    min_overall=cfg.confidence.autonomous_min_overall, quality_ok=True)
    # Every surviving claim is grounded and has RU + UZ (TZ §16 grounding = 100%).
    for c in rep.summary + rep.causal_explanation + rep.red_flags:
        assert c.grounded_by
        assert c.text_ru and c.text_uz
    assert rep.reasoning_trace  # trace is mandatory (TZ §9)


def test_grounding_drops_ungrounded_claims():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    data, _, _ = _pipeline_to_input(sig, "A")
    report = InterpretationReport(
        mode="A", confidence_value=0.5,
        summary=[
            Claim(text_ru="выдумка", text_uz="uydirma", grounded_by=["nonexistent_token"]),
            Claim(text_ru="без основания", text_uz="asossiz", grounded_by=[]),
        ],
    )
    res = validate_grounding(report, data)
    assert res.dropped == 2
    assert report.summary == []


def test_mode_B_escalates_on_critical():
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.5)
    data, saf, cfg = _pipeline_to_input(sig, "B")
    rep = interpret(data, saf, provider=StubProvider(),
                    min_overall=cfg.confidence.autonomous_min_overall, quality_ok=True)
    assert rep.mode == "escalate"
    assert rep.escalation_reason


def test_mode_B_autonomous_when_clean():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)  # plain normal
    data, saf, cfg = _pipeline_to_input(sig, "B")
    rep = interpret(data, saf, provider=StubProvider(),
                    min_overall=cfg.confidence.autonomous_min_overall, quality_ok=True)
    assert rep.mode == "B"


def test_mode_A_always_decision_support():
    sig = generate_synthetic_ecg(heart_rate_bpm=72, t_amp_uv=1200, potassium=6.5)
    data, saf, cfg = _pipeline_to_input(sig, "A")
    rep = interpret(data, saf, provider=StubProvider(),
                    min_overall=cfg.confidence.autonomous_min_overall, quality_ok=True)
    assert rep.mode == "A"  # stays A even with a critical finding
    assert rep.red_flags     # but red flags are shown
