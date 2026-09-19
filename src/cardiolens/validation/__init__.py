"""Validation harness and metrics (TZ §15, §16)."""

from cardiolens.validation.delineation import (
    DelineationReport,
    IntervalAccuracy,
    gold_intervals_ms,
    validate_ludb_delineation,
)
from cardiolens.validation.harness import (
    ECG_ARRHYTHMIA_SCORED_LABELS,
    PTBXL_SCORED_LABELS,
    PTBXL_SCP_TO_CODE,
    SCORED_LABELS,
    SNOMED_TO_CODE,
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
    "ECG_ARRHYTHMIA_SCORED_LABELS",
    "PTBXL_SCORED_LABELS",
    "PTBXL_SCP_TO_CODE",
    "SCORED_LABELS",
    "SNOMED_TO_CODE",
    "ClassMetrics",
    "DelineationReport",
    "IntervalAccuracy",
    "IntervalError",
    "LabeledCase",
    "ValidationReport",
    "build_ecg_arrhythmia_dataset",
    "build_ptbxl_dataset",
    "build_synthetic_dataset",
    "gold_intervals_ms",
    "parse_dx_from_header",
    "run_validation",
    "score_findings",
    "snomed_dx_to_codes",
    "validate_ludb_delineation",
]
