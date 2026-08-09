"""Layer 6 — interpretation & explanation (TZ §9).

``interpret`` orchestrates: pick a provider (Claude if available, else stub) ->
generate raw report -> validate grounding (drop ungrounded claims) -> apply the
A/B/escalate mode policy.
"""

from __future__ import annotations

from cardiolens.contracts.report import InterpretationReport
from cardiolens.layer6_interpret.claude import ClaudeProvider, available
from cardiolens.layer6_interpret.grounding import validate_grounding
from cardiolens.layer6_interpret.policy import decide_mode
from cardiolens.layer6_interpret.provider import InterpretationInput, LLMProvider
from cardiolens.layer6_interpret.stub import StubProvider
from cardiolens.safety.critical import SafetyResult


def default_provider(prefer_claude: bool = True) -> LLMProvider:
    if prefer_claude and available():
        return ClaudeProvider()
    return StubProvider()


def interpret(
    data: InterpretationInput,
    safety: SafetyResult,
    *,
    provider: LLMProvider | None = None,
    min_overall: float,
    quality_ok: bool,
) -> InterpretationReport:
    provider = provider or default_provider()
    report = provider.generate(data)

    validate_grounding(report, data)

    decision = decide_mode(
        data.mode, data.causal_graph, safety,
        min_overall=min_overall, quality_ok=quality_ok,
    )
    report.mode = decision.mode
    if decision.mode == "escalate":
        report.escalation_reason = decision.reason
        report.reasoning_trace.append(f"MODE B gate failed -> escalate: {decision.reason}")
    return report


__all__ = [
    "ClaudeProvider",
    "InterpretationInput",
    "LLMProvider",
    "StubProvider",
    "default_provider",
    "interpret",
]
