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
