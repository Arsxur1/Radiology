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

# "missed": the physician adds a finding the system did not report (a false negative;
# tracked for sensitivity, not used for confidence calibration).
Verdict = Literal["confirmed", "corrected", "rejected", "missed"]


class FindingFeedback(BaseModel):
    code: str
    verdict: Verdict
    # Confidence the calibrator is fitted on: the finding's confidence_base, filled in
    # by the server from the stored analysis when it is known (D87).
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence_shown: float | None = Field(default=None, ge=0.0, le=1.0)
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

    def summary(self) -> dict[str, dict]:
        """Per-finding verdict counts and observed precision (TZ §13, D87).

        precision = confirmed / (confirmed + corrected + rejected): how often the
        physician kept the finding as reported — the local counterpart of the PPV the
        D85/D86 confidence caps were set from.
        """
        out: dict[str, dict] = {}
        for fb in self.all():
            for f in fb.findings:
                row = out.setdefault(f.code, {"confirmed": 0, "corrected": 0, "rejected": 0,
                                              "missed": 0, "_conf": []})
                row[f.verdict] += 1
                if f.verdict != "missed":
                    shown = f.confidence_shown if f.confidence_shown is not None else f.confidence
                    row["_conf"].append(shown)
        for row in out.values():
            conf = row.pop("_conf")
            reported = row["confirmed"] + row["corrected"] + row["rejected"]
            row["precision"] = round(row["confirmed"] / reported, 3) if reported else None
            row["mean_confidence_shown"] = round(sum(conf) / len(conf), 3) if conf else None
        return dict(sorted(out.items()))

    def calibration_dataset(self) -> list[tuple[float, int]]:
        """Aggregate all feedback into calibrator training samples (TZ §13)."""
        samples: list[tuple[float, int]] = []
        for fb in self.all():
            samples.extend(fb.calibration_samples())
        return samples
