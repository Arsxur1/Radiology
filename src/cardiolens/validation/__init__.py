"""Validation harness and metrics (TZ §15, §16)."""

from cardiolens.validation.harness import (
    LabeledCase,
    ValidationReport,
    build_ecg_arrhythmia_dataset,
    build_ludb_dataset,
    build_ptbdb_dataset,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    filter_by_age,
    parse_dx_from_header,
    parse_patient_from_header,
    ptbdb_report_lines,
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
    "build_ludb_dataset",
    "build_ptbdb_dataset",
    "build_ptbxl_dataset",
    "build_synthetic_dataset",
    "filter_by_age",
    "parse_dx_from_header",
    "parse_patient_from_header",
    "ptbdb_report_lines",
    "run_validation",
    "score_findings",
    "snomed_dx_to_codes",
]
