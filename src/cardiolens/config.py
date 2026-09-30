"""Configuration loading and validation (TZ §0 — nothing hard-coded).

All runtime thresholds, norms and the pathophysiology rule base live in
``configs/`` and are loaded through here. Norm files are validated loosely (they
are reference tables); the rule base is validated against ``RuleBase``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from cardiolens.contracts.causal import RuleBase


def _repo_root() -> Path:
    # src/cardiolens/config.py -> repo root is three levels up.
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return _repo_root() / "configs"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class Disclaimers(BaseModel):
    ru: str
    uz: str


class ConfidenceGates(BaseModel):
    autonomous_min_overall: float
    autonomous_min_component: float


class Preprocessing(BaseModel):
    target_sampling_rate_hz: float
    bandpass_low_hz: float
    bandpass_high_hz: float
    sqi_accept_threshold: float
    # Longer recordings are analysed on their first max_analysis_s seconds (D37).
    max_analysis_s: float = 60.0


class DelineationCorrection(BaseModel):
    """Additive calibration (ms) for Layer-3 intervals — corrects the systematic
    fiducial bias of the neurokit2 delineator, measured against LUDB expert
    annotations (see DECISIONS.md D15a). Added to the measured interval; 0 = off.
    """
    pr_ms: float = 0.0
    qrs_ms: float = 0.0
    qt_ms: float = 0.0
    global_qt_ms: float = 0.0   # offset for the 12-lead median-beat QT (D30)
    global_pr_ms: float = 0.0   # offset for the 12-lead median-beat PR (D30)


class PauseConfig(BaseModel):
    """Pause / dropped-beat thresholds (TZ §7.1, §10)."""
    finding_ms: float = 2000.0
    ratio: float = 1.8
    critical_ms: float = 3000.0


class PacingConfig(BaseModel):
    """Paced-rhythm spike detection (TZ §7.1): an extreme slope outlier that is
    also time-locked to the QRS on at least coupling_min of the beats."""
    spike_ratio_min: float = 5.5
    coupling_min: float = 0.2


class SafetyConfig(BaseModel):
    """Critical-alert tiers (TZ §10). Long QT is a finding above the sex-specific
    QTc limit and a critical alert only from this QTc (D44)."""
    long_qt_critical_qtc_ms: float = 500.0


class QualityConfig(BaseModel):
    """Signal-organization quality gate (TZ §5). Below beat_consistency_min the
    QRS complexes are not self-consistent (artifact/noise), so the R-peak-derived
    findings are withheld rather than reported as confident garbage.

    A digitised image whose ``provenance.confidence`` is below
    digitized_confidence_min forces quality_ok low (mode B escalates).
    """
    beat_consistency_min: float = 0.8
    digitized_confidence_min: float = 0.8


class Versions(BaseModel):
    schema_version: str = Field(alias="schema")
    rules_base: str

    model_config = {"populate_by_name": True}


class AppConfig(BaseModel):
    mode: str
    mains_frequency_hz: float
    preprocessing: Preprocessing
    confidence: ConfidenceGates
    disclaimers: Disclaimers
    versions: Versions
    delineation_correction: DelineationCorrection = DelineationCorrection()
    pause: PauseConfig = PauseConfig()
    pacing: PacingConfig = PacingConfig()
    quality: QualityConfig = QualityConfig()
    safety: SafetyConfig = SafetyConfig()


@functools.lru_cache(maxsize=1)
def load_app_config() -> AppConfig:
    return AppConfig(**_load_yaml(configs_dir() / "app.yaml"))


@functools.lru_cache(maxsize=1)
def load_rule_base() -> RuleBase:
    return RuleBase(**_load_yaml(configs_dir() / "rules" / "pathophysiology.yaml"))


@functools.lru_cache(maxsize=2)
def load_norms(group: str) -> dict[str, Any]:
    """Load an age-group norm table ('adult' | 'pediatric')."""
    return _load_yaml(configs_dir() / "norms" / f"{group}.yaml")


def norms_for_patient(age_years: float | None, age_days: float | None) -> dict[str, Any]:
    """Pick the appropriate norm table for a patient (TZ §6).

    Children (age < 16y, or age given only in days) use pediatric norms.
    """
    is_pediatric = False
    if age_days is not None and age_years is None or age_years is not None and age_years < 16:
        is_pediatric = True
    return load_norms("pediatric" if is_pediatric else "adult")
