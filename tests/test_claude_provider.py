"""Claude provider against a fake ``anthropic`` SDK (no network): request shape, PHI,
grounding of the returned claims, refusal -> local stub (D91)."""

import json
import sys
import types

import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import claude
from cardiolens.layer6_interpret.claude import ClaudeProvider, _extract_json
from cardiolens.pipeline import run_pipeline


def _fake_sdk(monkeypatch, reply_text: str, stop_reason: str = "end_turn") -> dict:
    sent: dict = {}

    class _Messages:
        def create(self, **kw):
            sent.update(kw)
            block = types.SimpleNamespace(type="text", text=reply_text)
            return types.SimpleNamespace(stop_reason=stop_reason, content=[block])

    class _Anthropic:
        def __init__(self):
            self.beta = types.SimpleNamespace(messages=_Messages())

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))
    return sent


def _reply(grounded_by: list[str]) -> str:
    claim = {"text_ru": "Синусовый ритм.", "text_uz": "Sinus ritmi.", "grounded_by": grounded_by}
    body = {"summary": [claim, {"text_ru": "Выдумка.", "text_uz": "Uydirma.", "grounded_by": ["made_up"]}],
            "causal_explanation": [], "red_flags": [],
            "confidence_word_ru": "высокая", "confidence_word_uz": "yuqori",
            "next_step_ru": "-", "next_step_uz": "-", "reasoning_trace": ["ok"]}
    return "Here you go:\n" + json.dumps(body, ensure_ascii=False) + "\n"


def test_request_shape_phi_and_grounding(monkeypatch):
    sent = _fake_sdk(monkeypatch, _reply(["sinus_rhythm"]))
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    sig.patient.age_years = 54
    r = run_pipeline(sig, mode="A", provider=ClaudeProvider(model="test-model"))
    assert sent["model"] == "test-model" and sent["max_tokens"] == 16000
    assert sent["fallbacks"] == [{"model": "claude-opus-4-8"}]
    assert sent["output_config"] == {"effort": "medium"}
    data = json.loads(sent["messages"][0]["content"].split("DATA:\n", 1)[1])
    assert "patient" not in data and "context" not in data          # TZ §12: no demographics
    assert set(data["measurements"]) == {"heart_rate_bpm", "pr_ms", "qrs_ms", "qtc_bazett_ms"}
    texts = [c.text_ru for c in r.report.summary]
    assert "Синусовый ритм." in texts and "Выдумка." not in texts    # ungrounded claim dropped


def test_refusal_falls_back_to_the_local_stub(monkeypatch):
    _fake_sdk(monkeypatch, "", stop_reason="refusal")
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=72), mode="A", provider=ClaudeProvider())
    assert r.report.summary                                          # stub narrative present
    assert any("refusal" in line or "stub" in line.lower() for line in r.report.reasoning_trace)


def test_available_requires_key_and_package(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert claude.available() is False
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace())
    assert claude.available() is True


def test_model_override_from_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "env-model")
    assert ClaudeProvider().model == "env-model"


def test_extract_json_rejects_text_without_an_object():
    assert _extract_json('pre {"a": 1} post') == '{"a": 1}'
    with pytest.raises(ValueError):
        _extract_json("no json here")
