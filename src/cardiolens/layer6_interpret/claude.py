"""Anthropic Claude narrative provider for Layer 6 (TZ §9, §12).

Sends the causal graph + findings + measurements to Claude and requests a
schema-constrained bilingual (RU/UZ) interpretation. The model is instructed to
ground every claim in a provided token; the grounding validator enforces this
afterwards regardless of what the model returns (defence in depth, TZ §9).

PHI note (TZ §12): only derived features/labels are sent, never raw identifiers;
the payload passes ``phi.deidentify`` (allow-listed keys, identifier-shaped text
scrubbed) before it leaves the machine (D91).
Requires ``ANTHROPIC_API_KEY`` and the optional ``anthropic`` dependency; if
absent, callers should fall back to ``StubProvider``. A refusal or any API error
raises, and ``interpret`` falls back to the deterministic stub.
"""

from __future__ import annotations

import json
import os

from cardiolens.contracts.report import Claim, InterpretationReport
from cardiolens.layer6_interpret.phi import deidentify
from cardiolens.layer6_interpret.provider import InterpretationInput, LLMProvider

_SYSTEM = (
    "You are a cardiology ECG interpretation assistant. You are given structured "
    "findings and a causal graph derived by a deterministic engine. Write a "
    "bilingual (Russian + Uzbek) interpretation. CRITICAL RULES: (1) Do NOT invent "
    "findings or causal links absent from the provided data. (2) Every claim must "
    "cite a grounding token from the provided list in 'grounded_by'. (3) Output "
    "ONLY valid JSON matching the requested schema. (4) Provide a visible "
    "reasoning trace."
)


def available() -> bool:
    """True if a Claude call is possible in this environment."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, model: str | None = None, max_tokens: int = 16000):
        # Model is configurable; defaults to the current Opus, overridable via the
        # ANTHROPIC_MODEL environment variable. Thinking is always on for it and
        # counts against max_tokens, so the budget leaves room for the JSON answer.
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.max_tokens = max_tokens
        self.redactions = 0

    def _payload(self, data: InterpretationInput) -> dict:
        return {
            "mode": data.mode,
            "grounding_tokens": sorted(data.grounding_tokens()),
            "findings": [
                {"code": f.code, "ru": f.label_ru, "uz": f.label_uz,
                 "confidence": f.confidence,
                 "evidence": [f"{e.feature}={e.value}" for e in f.evidence]}
                for f in data.findings.findings
            ],
            "causal_edges": [
                {"id": f"{e.from_}->{e.to}", "relation": e.relation}
                for e in data.causal_graph.edges
            ],
            "critical_flags": data.causal_graph.critical_flags,
            "implausible_flags": data.causal_graph.implausible_flags,
            "confidence": data.causal_graph.confidence,
            "measurements": {
                "heart_rate_bpm": data.measurements.heart_rate_bpm,
                "pr_ms": data.measurements.intervals.pr_ms,
                "qrs_ms": data.measurements.intervals.qrs_ms,
                "qtc_bazett_ms": data.measurements.intervals.qtc_bazett_ms,
            },
        }

    def generate(self, data: InterpretationInput) -> InterpretationReport:
        import anthropic

        client = anthropic.Anthropic()
        payload, self.redactions = deidentify(self._payload(data))
        prompt = (
            "Interpret this ECG. Return JSON with keys: summary[], "
            "causal_explanation[], differential[], red_flags[], "
            "confidence_word_ru, confidence_word_uz, next_step_ru, next_step_uz, "
            "reasoning_trace[]. Each summary/causal/red_flag item is "
            "{text_ru, text_uz, grounded_by[]}.\n\nDATA:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
        )
        # A policy decline is re-run server-side on the fallback model inside the
        # same call; a refusal of the whole chain raises and interpret() falls back
        # to the deterministic stub.
        msg = client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=_SYSTEM,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-06-01"],
            fallbacks=[{"model": "claude-opus-4-8"}],
            messages=[{"role": "user", "content": prompt}],
        )
        if msg.stop_reason == "refusal":
            raise RuntimeError("Claude declined the request (stop_reason=refusal)")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        parsed = json.loads(_extract_json(text))

        def claims(key: str) -> list[Claim]:
            return [Claim(**c) for c in parsed.get(key, [])]

        return InterpretationReport(
            mode=data.mode,
            summary=claims("summary"),
            causal_explanation=claims("causal_explanation"),
            red_flags=claims("red_flags"),
            differential=[],  # kept deterministic from the engine
            confidence_value=data.causal_graph.confidence,
            confidence_word_ru=parsed.get("confidence_word_ru", ""),
            confidence_word_uz=parsed.get("confidence_word_uz", ""),
            next_step_ru=parsed.get("next_step_ru", ""),
            next_step_uz=parsed.get("next_step_uz", ""),
            reasoning_trace=parsed.get("reasoning_trace", []),
            disclaimer_ru=data.disclaimer_ru,
            disclaimer_uz=data.disclaimer_uz,
        )


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    return text[start : end + 1]
