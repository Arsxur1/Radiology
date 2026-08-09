"""LLM provider abstraction for Layer 6 (TZ §9, §12).

Everything Layer 6 needs to produce a narrative is bundled in
``InterpretationInput``. Providers return a raw ``InterpretationReport`` that is
then passed through the grounding validator and mode policy. Two providers ship:
``StubProvider`` (deterministic, offline — for tests and full-offline runs) and
``ClaudeProvider`` (Anthropic cloud narrative).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from cardiolens.contracts.causal import CausalGraph
from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import FindingSet
from cardiolens.contracts.report import InterpretationReport
from cardiolens.contracts.signal import Patient, PatientContext


@dataclass
class InterpretationInput:
    """Grounding material handed to the LLM (TZ §9 "Вход LLM")."""

    findings: FindingSet
    causal_graph: CausalGraph
    measurements: Measurements
    patient: Patient
    context: PatientContext
    mode: str                     # requested mode "A" | "B"
    critical_labels: list[tuple[str, str]]   # (ru, uz) from safety layer
    disclaimer_ru: str
    disclaimer_uz: str

    def grounding_tokens(self) -> set[str]:
        """Valid tokens a claim may cite: feature names, finding codes, edge ids."""
        tokens: set[str] = set()
        for f in self.findings.findings:
            tokens.add(f.code)
            for e in f.evidence:
                tokens.add(e.feature)
        for edge in self.causal_graph.edges:
            tokens.add(f"{edge.from_}->{edge.to}")
        for n in self.causal_graph.nodes:
            tokens.add(n.id)
        # Activated causal rules and critical flags are valid grounds too:
        # a red flag may cite the critical rule / finding that raised it.
        tokens.update(self.causal_graph.activated_rules)
        tokens.update(self.causal_graph.critical_flags)
        return tokens


class LLMProvider(abc.ABC):
    """Base class for Layer-6 narrative providers."""

    name: str = "base"

    @abc.abstractmethod
    def generate(self, data: InterpretationInput) -> InterpretationReport:
        """Produce a raw (pre-grounding-validation) interpretation report."""
        raise NotImplementedError
