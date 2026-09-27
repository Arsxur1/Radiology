"""Local OpenAI-compatible LLM provider (TZ §12, §17): mocked server, fallback, selection."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cardiolens.config import LLMConfig, load_app_config
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import (
    LocalOpenAIProvider,
    StubProvider,
    default_provider,
    local_available,
)
from cardiolens.pipeline import run_pipeline


class _Handler(BaseHTTPRequestHandler):
    """Answers /models and /chat/completions with a canned grounded/ungrounded mix."""

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        self.send_response(200 if self.path.endswith("/models") else 404)
        self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"data": []}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n))
        self.server.last_request = req
        content = json.dumps({
            "summary": [{"text_ru": "Синусовый ритм", "text_uz": "Sinus ritmi", "grounded_by": ["sinus_rhythm"]},
                        {"text_ru": "Выдумка", "text_uz": "Uydirma", "grounded_by": ["not_a_real_token"]}],
            "causal_explanation": [], "differential": [], "red_flags": [],
            "confidence_word_ru": "высокая", "confidence_word_uz": "yuqori",
            "next_step_ru": "Наблюдение", "next_step_uz": "Kuzatuv", "reasoning_trace": ["mock"],
        }, ensure_ascii=False)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", srv
    srv.shutdown()


def test_local_provider_end_to_end_with_grounding(mock_server):
    url, srv = mock_server
    provider = LocalOpenAIProvider(url, model="mock-model", timeout_s=5)
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=75, seed=9), mode="A", provider=provider)
    texts = [c.text_ru for c in r.report.summary]
    assert "Синусовый ритм" in texts
    assert "Выдумка" not in texts                      # ungrounded claim dropped by the validator
    assert any("local model mock-model" in t for t in r.report.reasoning_trace)
    assert srv.last_request["model"] == "mock-model"
    assert "grounding_tokens" in srv.last_request["messages"][1]["content"]


def test_local_provider_falls_back_to_stub_when_unreachable():
    provider = LocalOpenAIProvider("http://127.0.0.1:9/v1", timeout_s=1)
    r = run_pipeline(generate_synthetic_ecg(heart_rate_bpm=75, seed=9), mode="A", provider=provider)
    assert r.report.summary                            # deterministic narrative still produced
    assert any("local LLM unavailable" in t for t in r.report.reasoning_trace)
    assert local_available("http://127.0.0.1:9/v1") is False


def test_provider_selection_from_config(mock_server, monkeypatch):
    url, _ = mock_server
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(default_provider(llm=LLMConfig(provider="stub")), StubProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="local", local_base_url=url)), LocalOpenAIProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="auto", local_base_url=url)), LocalOpenAIProvider)
    assert isinstance(default_provider(llm=LLMConfig(provider="auto", local_base_url="http://127.0.0.1:9/v1")), StubProvider)
    assert load_app_config().llm.provider == "auto"
