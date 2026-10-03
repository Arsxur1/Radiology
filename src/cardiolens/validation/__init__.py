"""Validation harness and metrics (TZ §15, §16)."""

from cardiolens.validation.harness import (
    SCORED_LABELS,
    LabeledCase,
    ValidationReport,
    build_ecg_arrhythmia_dataset,
    build_ludb_dataset,
    build_mimic_dataset,
    build_ptbdb_dataset,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    filter_by_age,
    mimic_statements_to_codes,
    parse_dx_from_header,
    parse_patient_from_header,
    ptbdb_report_lines,
    run_validation,
    snomed_dx_to_codes,
)
from cardiolens.validation.metrics import ClassMetrics, IntervalError, score_findings
from cardiolens.validation.risk import TierMetrics, risk_report_lines, score_risk_tiers

__all__ = [
    "SCORED_LABELS",
    "ClassMetrics",
    "IntervalError",
    "LabeledCase",
    "TierMetrics",
    "ValidationReport",
    "build_ecg_arrhythmia_dataset",
    "build_ludb_dataset",
    "build_mimic_dataset",
    "build_ptbdb_dataset",
    "build_ptbxl_dataset",
    "build_synthetic_dataset",
    "filter_by_age",
    "mimic_statements_to_codes",
    "parse_dx_from_header",
    "parse_patient_from_header",
    "ptbdb_report_lines",
    "risk_report_lines",
    "run_validation",
    "score_findings",
    "score_risk_tiers",
    "snomed_dx_to_codes",
]
