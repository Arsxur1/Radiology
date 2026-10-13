"""Interpretation report contract — Layer 6 output (TZ §9, §11).

The LLM output is schema-constrained: every claim must ground back to a feature
(evidence) or a causal-graph edge, otherwise it is dropped (TZ §9 "Заземление").
All narrative fields are bilingual (RU + UZ) per TZ §0.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["A", "B", "escalate"]


class Claim(BaseModel):
    """A single grounded statement (TZ §9).

    ``grounded_by`` references either a feature name (Layer 3/4 evidence) or an
    edge id ``"<from>-><to>"`` from the causal graph. A claim with no grounding
    is rejected by the grounding validator.
    """

    text_ru: str
    text_uz: str
    grounded_by: list[str] = Field(default_factory=list)


class Differential(BaseModel):
    label_ru: str
    label_uz: str
    priority: int          # 1 = most likely
    confidence: float = Field(ge=0.0, le=1.0)


class InterpretationReport(BaseModel):
    """Full Layer-6 output (TZ §9)."""

    mode: Mode                                  # A | B | escalate
    summary: list[Claim] = Field(default_factory=list)          # (a) резюме находок
    causal_explanation: list[Claim] = Field(default_factory=list)  # (b) цепочка
    differential: list[Differential] = Field(default_factory=list)  # (c)
    confidence_value: float = Field(ge=0.0, le=1.0)             # (d)
    confidence_word_ru: str = ""
    confidence_word_uz: str = ""
    red_flags: list[Claim] = Field(default_factory=list)        # (e) критические
    next_step_ru: str = ""                                      # (f) следующий шаг
    next_step_uz: str = ""
    # Reasoning trace is mandatory and user-visible (TZ §9).
    reasoning_trace: list[str] = Field(default_factory=list)
    # Who wrote the narrative: "stub" | "claude" | "local" (TZ §12 — visible in the UI/JSON).
    provider: str = "stub"
    provider_model: str | None = None
    escalation_reason: str | None = None       # set when mode == "escalate"
    disclaimer_ru: str = ""
    disclaimer_uz: str = ""

    def all_claims(self) -> list[Claim]:
        return [
            *self.summary,
            *self.causal_explanation,
            *self.red_flags,
        ]
