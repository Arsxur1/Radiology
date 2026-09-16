"""Validation harness and metrics (TZ §15, §16)."""

from cardiolens.validation.harness import (
    LabeledCase,
    ValidationReport,
    build_ecg_arrhythmia_dataset,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    parse_dx_from_header,
    run_validation,
    snomed_dx_to_codes,
)
from cardiolens.validation.metrics import ClassMetrics, IntervalError, score_findings

__all__ = [
    "ClassMetrics",
    "IntervalError",
    "LabeledCase",
    "ValidationReport",
    "build_ecg_arrhythmia_dataset",
    "build_ptbxl_dataset",
    "build_synthetic_dataset",
    "parse_dx_from_header",
    "run_validation",
    "score_findings",
    "snomed_dx_to_codes",
]
