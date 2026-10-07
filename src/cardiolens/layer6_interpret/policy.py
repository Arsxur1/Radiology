"""Mode policy for autonomous vs decision-support output (TZ §2, §9, §10).

Mode A (decision support) always emits a report for the physician. Mode B
(autonomous) emits an assertive report ONLY when every gate passes; otherwise it
escalates to "требуется просмотр врача" with a reason. A critical finding never
permits autonomous output (TZ §10).
"""

from __future__ import annotations

from dataclasses import dataclass

from cardiolens.contracts.causal import CausalGraph
from cardiolens.contracts.findings import FindingSet
from cardiolens.safety.critical import SafetyResult


@dataclass
class ModeDecision:
    mode: str                 # "A" | "B" | "escalate"
    reason: str | None = None


def decide_mode(
    requested_mode: str,
    causal_graph: CausalGraph,
    safety: SafetyResult,
    *,
    min_overall: float,
    quality_ok: bool,
    findings: FindingSet | None = None,
    min_component: float | None = None,
    stemi_j_criterion: bool | None = None,
) -> ModeDecision:
    """Apply the §2 gating logic."""
    if requested_mode != "B":
        return ModeDecision(mode="A")

    # Mode B requested — check every gate (TZ §2).
    if safety.has_critical or causal_graph.critical_flags:
        return ModeDecision(
            mode="escalate",
            reason="критическая находка требует просмотра врача (§10)",
        )
    if causal_graph.implausible_flags:
        return ModeDecision(
            mode="escalate",
            reason="физиологически неправдоподобные находки — возможен артефакт/"
                   "перестановка электродов (§8.3)",
        )
    # The ST-elevation finding reads J+60; the guideline (UDMI) STEMI criterion reads the
    # J point. A J-point elevation that meets it without the finding (fast ST descent,
    # e.g. early repolarisation) raises no alert, but must not be reported autonomously
    # either (D92: 2 of 4 autonomous ST-elevation-labelled ECGs on fresh Georgia).
    if stemi_j_criterion:
        return ModeDecision(
            mode="escalate",
            reason="подъём точки J соответствует критерию STEMI (UDMI) — нужен просмотр врача (§10)",
        )
    if not quality_ok:
        return ModeDecision(
            mode="escalate",
            reason="качество сигнала ниже порога (§2)",
        )
    if causal_graph.confidence < min_overall:
        return ModeDecision(
            mode="escalate",
            reason=f"общая уверенность {causal_graph.confidence} < порога {min_overall} (§2)",
        )
    # Per-finding floor (D85): one weak finding must not ride on a confident mean
    # (sinus rhythm 0.85 + a 0.60 finding still averages near the overall gate).
    if findings is not None and min_component is not None:
        weak = [f for f in findings.findings if f.confidence < min_component]
        if weak:
            listed = ", ".join(f"{f.code} {f.confidence}" for f in weak)
            return ModeDecision(
                mode="escalate",
                reason=f"уверенность находки ниже порога {min_component}: {listed} (§2)",
            )
    return ModeDecision(mode="B")
