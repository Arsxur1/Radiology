"""Validation harness and metrics (TZ §15, §16)."""

from cardiolens.validation.harness import (
    LabeledCase,
    ValidationReport,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    run_validation,
)
from cardiolens.validation.metrics import ClassMetrics, IntervalError, score_findings

__all__ = [
    "ClassMetrics",
    "IntervalError",
    "LabeledCase",
    "ValidationReport",
    "build_ptbxl_dataset",
    "build_synthetic_dataset",
    "run_validation",
    "score_findings",
]
