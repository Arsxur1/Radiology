"""Learnable layer on top of the rule-based findings (TZ §17 v2).

A gradient-boosted model over Layer-3 measurements gives a second opinion per
finding class. It never replaces the rules or the safety layer: Layer 4 uses
its probabilities as extra evidence (see ``layer4_findings.ml``).
"""

from cardiolens.ml.features import FEATURE_NAMES, features_from_result
from cardiolens.ml.model import RhythmModel, load_model

__all__ = ["FEATURE_NAMES", "RhythmModel", "features_from_result", "load_model"]
