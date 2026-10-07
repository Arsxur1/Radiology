"""De-identification of the cloud-LLM payload (TZ §12, D91)."""

import json

import pytest

from cardiolens.api.service import DEMO_CASES, signal_from_case
from cardiolens.contracts.findings import Evidence, Finding
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import interpret
from cardiolens.layer6_interpret.claude import ClaudeProvider
from cardiolens.layer6_interpret.phi import REDACTED, deidentify, scrub_text
from cardiolens.layer6_interpret.provider import LLMProvider
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.pipeline import run_pipeline


class _Capture(LLMProvider):
    """Records what a cloud provider would be given, answers with the stub."""

    name = "capture"

    def __init__(self):
        self.data = None

    def generate(self, data):
        self.data = data
        return StubProvider().generate(data)


def _payload_for(sig):
    cap = _Capture()
    run_pipeline(sig, mode="A", provider=cap)
    return ClaudeProvider()._payload(cap.data), cap.data


@pytest.mark.parametrize("case", sorted(DEMO_CASES))
def test_real_payloads_need_no_redaction(case):
    # Clinical text and numbers must pass untouched (no over-redaction).
    payload, _ = _payload_for(signal_from_case(case))
    clean, n = deidentify(payload)
    assert n == 0 and clean == payload


def test_patient_context_never_reaches_the_payload():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.context.symptoms = ["Иванов Иван, тел. +998 90 123 45 67"]
    sig.context.medications = ["назначил др. Петров 12.03.2024"]
    sig.context.prior_ecg_id = "MRN 12345678"
    sig.patient.age_years = 61
    payload, _ = _payload_for(sig)
    text = json.dumps(payload, ensure_ascii=False)
    for leak in ("Иванов", "123 45 67", "Петров", "12.03.2024", "12345678"):
        assert leak not in text
    assert set(payload) <= {"mode", "grounding_tokens", "findings", "causal_edges",
                            "critical_flags", "implausible_flags", "confidence",
                            "measurements"}


def test_identifiers_in_findings_are_scrubbed_before_the_cloud_call():
    _, data = _payload_for(generate_synthetic_ecg(heart_rate_bpm=72))
    data.findings.findings.append(Finding(
        code="x", label_ru="пациент ivanov@mail.uz, д.р. 12.03.1961, паспорт AA1234567",
        label_uz="tel +998 (90) 123-45-67", confidence=0.5,
        evidence=[Evidence(feature="record", value="MRN 0012345678")]))
    clean, n = deidentify(ClaudeProvider()._payload(data))
    text = json.dumps(clean, ensure_ascii=False)
    assert n >= 5
    for leak in ("ivanov@mail.uz", "12.03.1961", "AA1234567", "123-45-67", "0012345678"):
        assert leak not in text
    assert REDACTED in text


def test_ecg_numbers_are_not_mistaken_for_identifiers():
    for s in ("448.0/5.0/-28.0/-273.0", "ЧСС 74.1/мин", "II,aVF,V3", "-28.0 -273.0 -150.0",
              "QTc 512 ms", "QRS 120 мс; PR 210 мс", "ось -45°"):
        assert scrub_text(s) == (s, 0)


def test_unknown_top_level_keys_are_dropped():
    clean, n = deidentify({"mode": "A", "patient": {"name": "Иванов"}})
    assert clean == {"mode": "A"} and n == 1


def test_cloud_failure_falls_back_to_local_narrative():
    class _Failing(LLMProvider):
        name = "claude"

        def generate(self, data):
            raise RuntimeError("Claude declined the request (stop_reason=refusal)")

    cap = _Capture()
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A", provider=cap)
    report = interpret(cap.data, r.safety, provider=_Failing(),
                       min_overall=r.config.confidence.autonomous_min_overall,
                       quality_ok=True)
    assert report.summary
    assert any("local stub narrative" in t for t in report.reasoning_trace)
