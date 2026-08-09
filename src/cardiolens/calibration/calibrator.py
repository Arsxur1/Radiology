"""Confidence calibration via temperature scaling (TZ §13, §16).

A raw confidence of 0.9 should mean the finding is correct ~90% of the time —
critical for the mode-B gating thresholds (TZ §2). We fit a single scalar
temperature ``T`` on the model's logits so that scaled probabilities match
observed accuracy, then report calibration error (ECE, TZ §16).

Fitting needs labelled outcomes (physician-confirmed findings, TZ §13). Until
enough are collected, ``TemperatureScaler`` defaults to identity (T=1) and is a
no-op — confidence passes through unchanged.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class TemperatureScaler:
    """Single-parameter temperature scaler for probability calibration."""

    temperature: float = 1.0
    fitted: bool = False

    def apply(self, confidence: float) -> float:
        """Scale one confidence value. Identity when unfitted (T=1)."""
        return _sigmoid(_logit(confidence) / self.temperature)

    def fit(self, samples: list[tuple[float, int]], *, max_iter: int = 200,
            lr: float = 0.05) -> TemperatureScaler:
        """Fit T on (confidence, label) pairs by minimising NLL (grad descent).

        ``label`` is 1 if the finding was correct, else 0.
        """
        if len(samples) < 10:
            # Too few labels to calibrate reliably — stay identity (TZ §0/§13).
            self.temperature, self.fitted = 1.0, False
            return self
        logits = [_logit(c) for c, _ in samples]
        labels = [float(y) for _, y in samples]
        t = 1.0
        for _ in range(max_iter):
            grad = 0.0
            for z, y in zip(logits, labels):
                p = _sigmoid(z / t)
                # d NLL / d T
                grad += (p - y) * (-z / (t * t))
            grad /= len(samples)
            t -= lr * grad
            t = max(0.05, min(10.0, t))
        self.temperature, self.fitted = float(t), True
        return self

    def to_dict(self) -> dict:
        return {"temperature": self.temperature, "fitted": self.fitted}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(temperature=float(d.get("temperature", 1.0)),
                   fitted=bool(d.get("fitted", False)))


def expected_calibration_error(samples: list[tuple[float, int]], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) — acceptance metric for mode B (TZ §16)."""
    if not samples:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for conf, label in samples:
        idx = min(n_bins - 1, int(conf * n_bins))
        bins[idx].append((conf, label))
    total = len(samples)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(avg_conf - acc)
    return round(ece, 4)
