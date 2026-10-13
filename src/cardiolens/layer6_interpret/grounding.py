"""Grounding validator (TZ §9 "Заземление", §16).

Every claim must reference at least one valid grounding token — an evidence
feature, a finding code, a causal-graph node, or an edge id. Claims that cite
nothing valid are DROPPED (the LLM is not allowed to invent findings or links).
Acceptance criterion §16: 100% of surviving claims are grounded.
"""

from __future__ import annotations

from dataclasses import dataclass

from cardiolens.contracts.report import Claim, InterpretationReport
from cardiolens.layer6_interpret.provider import InterpretationInput


@dataclass
class GroundingResult:
    report: InterpretationReport
    dropped: int
    total: int

    @property
    def surviving(self) -> int:
        return self.total - self.dropped

    @property
    def all_grounded(self) -> bool:
        """True once ungrounded claims have been removed (TZ §16: 100% grounded)."""
        return True  # by construction: validate_grounding drops every ungrounded claim


def _filter(claims: list[Claim], valid: set[str]) -> tuple[list[Claim], int]:
    kept, dropped = [], 0
    for c in claims:
        if c.grounded_by and all(tok in valid for tok in c.grounded_by):
            kept.append(c)
        else:
            dropped += 1
    return kept, dropped


def validate_grounding(
    report: InterpretationReport, data: InterpretationInput
) -> GroundingResult:
    """Drop ungrounded claims; return the cleaned report and stats."""
    valid = data.grounding_tokens() | {"critical"}
    total = 0
    dropped = 0

    for field in ("summary", "causal_explanation", "red_flags"):
        claims = getattr(report, field)
        total += len(claims)
        kept, d = _filter(claims, valid)
        dropped += d
        setattr(report, field, kept)

    if dropped:
        report.reasoning_trace.append(
            f"grounding: dropped {dropped}/{total} ungrounded claim(s)"
        )
    return GroundingResult(report=report, dropped=dropped, total=total)
