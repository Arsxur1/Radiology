"""Physician feedback capture (TZ §13).

Captures structured physician corrections — for each finding/chain: confirmed,
corrected, or rejected — into an append-only JSONL log. This is the fuel for
future fine-tuning and, crucially, for fitting the confidence calibrator
(TZ §13): each confirmed/rejected finding is a (confidence, label) sample.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["confirmed", "corrected", "rejected"]


class FindingFeedback(BaseModel):
    code: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)   # model confidence at review time
    corrected_code: str | None = None            # set when verdict == "corrected"
    comment: str | None = None


class Feedback(BaseModel):
    """One physician review of an analysis (TZ §13)."""

    analysis_id: str
    reviewer: str | None = None
    mode: str | None = None                      # mode the system was in
    findings: list[FindingFeedback] = Field(default_factory=list)
    overall_agreement: bool | None = None
    timestamp: float = Field(default_factory=lambda: time.time())

    def calibration_samples(self) -> list[tuple[float, int]]:
        """(confidence, label) pairs for calibrator fitting (TZ §13, §16).

        confirmed -> 1, rejected -> 0. 'corrected' is ambiguous and skipped.
        """
        out: list[tuple[float, int]] = []
        for f in self.findings:
            if f.verdict == "confirmed":
                out.append((f.confidence, 1))
            elif f.verdict == "rejected":
                out.append((f.confidence, 0))
        return out


@dataclass
class FeedbackStore:
    """Append-only JSONL feedback log."""

    path: Path
    _cache: list[Feedback] = field(default_factory=list)

    @classmethod
    def at(cls, path: str | Path) -> FeedbackStore:
        return cls(path=Path(path))

    def append(self, fb: Feedback) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fb.model_dump(), ensure_ascii=False) + "\n")

    def all(self) -> list[Feedback]:
        if not self.path.exists():
            return []
        out: list[Feedback] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Feedback(**json.loads(line)))
        return out

    def calibration_dataset(self) -> list[tuple[float, int]]:
        """Aggregate all feedback into calibrator training samples (TZ §13)."""
        samples: list[tuple[float, int]] = []
        for fb in self.all():
            samples.extend(fb.calibration_samples())
        return samples
