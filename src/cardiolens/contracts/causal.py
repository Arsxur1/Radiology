"""Causal analysis contracts — Layer 5 rule schema and output graph (TZ §8).

Two concerns live here:
  * ``RuleBase`` / ``Rule`` — the schema that validates
    ``configs/rules/pathophysiology.yaml`` (declarative causal knowledge).
  * ``CausalGraph`` — the structured output of Layer 5 (nodes, edges, labels,
    critical / implausibility flags) that Layer 6 turns into narrative.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Physiologic = Literal["physiologic", "pathologic"]
NodeKind = Literal["cause", "mechanism", "finding", "artifact"]
Relation = Literal["causes", "progresses_to", "reciprocal", "explained_by"]


# --------------------------------------------------------------------------- #
# Rule-base schema (validates the YAML knowledge base)
# --------------------------------------------------------------------------- #
class RuleNode(BaseModel):
    id: str
    kind: NodeKind
    label_ru: str
    label_uz: str


class RuleEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    relation: Relation

    model_config = {"populate_by_name": True}


class ContextSupport(BaseModel):
    potassium_min: float | None = None
    potassium_max: float | None = None
    calcium_min: float | None = None
    calcium_max: float | None = None


class Rule(BaseModel):
    id: str
    label_ru: str
    label_uz: str
    trigger_findings: list[str]
    context_support: ContextSupport | None = None
    physiologic: Physiologic
    critical: bool = False
    nodes: list[RuleNode]
    edges: list[RuleEdge]
    explanation_ru: str
    explanation_uz: str


class RuleBase(BaseModel):
    version: str
    rules: list[Rule]


# --------------------------------------------------------------------------- #
# Causal-graph output (Layer 5 -> Layer 6)
# --------------------------------------------------------------------------- #
class GraphNode(BaseModel):
    id: str
    kind: NodeKind
    label_ru: str
    label_uz: str
    physiologic: Physiologic
    observed: bool = False   # was this finding actually measured on the ECG?


class GraphEdge(BaseModel):
    from_: str = Field(alias="from")
    to: str
    relation: Relation

    model_config = {"populate_by_name": True}


class CausalGraph(BaseModel):
    """Layer-5 output graph (TZ §8 final output)."""

    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    activated_rules: list[str] = Field(default_factory=list)
    critical_flags: list[str] = Field(default_factory=list)
    # Findings no mechanism can explain -> "physiologically implausible" (TZ §8.3).
    implausible_flags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}

    def edge_pairs(self) -> set[tuple[str, str]]:
        return {(e.from_, e.to) for e in self.edges}
