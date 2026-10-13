"""Validation metrics (TZ §16).

Per-class sensitivity/specificity for finding detection, and mean absolute error
for interval measurements. Data-source agnostic: the harness feeds gold labels;
these functions just score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassMetrics:
    label: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def sensitivity(self) -> float | None:
        d = self.tp + self.fn
        return round(self.tp / d, 3) if d else None

    @property
    def specificity(self) -> float | None:
        d = self.tn + self.fp
        return round(self.tn / d, 3) if d else None

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def to_dict(self) -> dict:
        return {"label": self.label, "tp": self.tp, "fp": self.fp,
                "tn": self.tn, "fn": self.fn, "support": self.support,
                "sensitivity": self.sensitivity, "specificity": self.specificity}


def score_findings(
    predictions: list[set[str]],
    gold: list[set[str]],
    labels: list[str],
) -> dict[str, ClassMetrics]:
    """Confusion counts per label across a dataset (TZ §16 sens/spec)."""
    out = {lbl: ClassMetrics(label=lbl) for lbl in labels}
    for pred, truth in zip(predictions, gold):
        for lbl in labels:
            p, t = lbl in pred, lbl in truth
            m = out[lbl]
            if p and t:
                m.tp += 1
            elif p and not t:
                m.fp += 1
            elif not p and t:
                m.fn += 1
            else:
                m.tn += 1
    return out


@dataclass
class IntervalError:
    name: str
    errors: list[float] = field(default_factory=list)

    def add(self, predicted: float | None, gold: float | None) -> None:
        if predicted is not None and gold is not None:
            self.errors.append(abs(predicted - gold))

    @property
    def mae(self) -> float | None:
        return round(sum(self.errors) / len(self.errors), 2) if self.errors else None

    @property
    def n(self) -> int:
        return len(self.errors)

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "mae_ms": self.mae}


def macro_recall(metrics: dict[str, ClassMetrics]) -> float | None:
    vals = [m.sensitivity for m in metrics.values() if m.sensitivity is not None]
    return round(sum(vals) / len(vals), 3) if vals else None
