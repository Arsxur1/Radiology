"""Learnable layer (TZ §17 v2): evidence only, safety untouched, graceful without artifact."""

import numpy as np
import pytest

from cardiolens.config import load_app_config
from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer4_findings.ml import (
    AGREE_P,
    CONFIDENCE_CAP,
    SUGGEST_PREFIX,
    apply_ml_evidence,
)
from cardiolens.ml.features import FEATURE_NAMES, features_from_measurements
from cardiolens.ml.model import fit_model, load_model
from cardiolens.pipeline import run_pipeline
from cardiolens.safety import scan_critical


def _finding(code: str, conf: float = 0.7) -> Finding:
    return Finding(code=code, group="conduction", label_ru=code, label_uz=code,
                   evidence=[Evidence(feature="x", value="1")], confidence=conf)


def _measurements(probs) -> Measurements:
    m = Measurements(heart_rate_bpm=70.0)
    m.ml_probabilities = probs
    return m


def test_agreeing_probability_bumps_confidence_and_records_evidence():
    fs = FindingSet(findings=[_finding("rbbb", 0.9), _finding("lbbb", 0.7)])
    audit = apply_ml_evidence(fs, _measurements({"rbbb": 0.95, "lbbb": 0.05}))
    rbbb, lbbb = fs.findings
    assert rbbb.confidence == CONFIDENCE_CAP                  # capped
    assert lbbb.confidence == 0.7                             # dissent never lowers
    assert any(e.feature == "ml_p" for e in lbbb.evidence)
    assert any("doubts lbbb" in a for a in audit)


def test_suggestion_only_for_allowed_classes_without_rule():
    fs = FindingSet(findings=[_finding("rbbb")])
    probs = {"rbbb": 0.99, "lbbb": 0.9, "lvh": 0.99}
    apply_ml_evidence(fs, _measurements(probs), suggest_classes=["rbbb", "lbbb"])
    codes = fs.codes()
    assert SUGGEST_PREFIX + "lbbb" in codes        # allowed, no rule, p >= SUGGEST_P
    assert SUGGEST_PREFIX + "rbbb" not in codes    # rule already fired
    assert SUGGEST_PREFIX + "lvh" not in codes     # not an allowed class
    assert all(not f.critical for f in fs.findings if f.code.startswith(SUGGEST_PREFIX))


def test_safety_scan_ignores_ml_suggestions():
    fs = FindingSet(findings=[])
    apply_ml_evidence(fs, _measurements({"st_elevation": 0.99}), suggest_classes=["st_elevation"])
    assert fs.codes() == {SUGGEST_PREFIX + "st_elevation"}
    assert scan_critical(fs, Measurements(heart_rate_bpm=70.0)).alerts == []


def test_no_probabilities_is_a_no_op():
    fs = FindingSet(findings=[_finding("rbbb")])
    assert apply_ml_evidence(fs, Measurements(heart_rate_bpm=70.0)) == []
    assert fs.findings[0].confidence == 0.7


def test_pipeline_runs_without_artifact_and_with_ml_disabled(tmp_path):
    cfg = load_app_config()
    sig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0)
    missing = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"model_path": str(tmp_path / "none.joblib")})})
    r1 = run_pipeline(sig, mode="A", config=missing)
    assert r1.measurements.ml_probabilities is None
    off = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"enabled": False})})
    r2 = run_pipeline(sig, mode="A", config=off)
    assert r2.measurements.ml_probabilities is None
    assert not any(a.startswith("[ML]") for a in r2.audit_log)


def test_tiny_model_end_to_end(tmp_path):
    """Fit a toy model in-test and check it flows through the pipeline as evidence."""
    rng = np.random.default_rng(0)
    sig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=10.0)
    m = run_pipeline(sig, mode="A").measurements
    x0 = features_from_measurements(m, sig.sampling_rate_hz)
    n = 80
    x = np.nan_to_num(np.repeat(x0[None, :], n, axis=0)) + rng.normal(0, 0.01, (n, len(FEATURE_NAMES)))
    # Half the rows get a shifted heart rate feature and the positive label.
    y = np.zeros(n, dtype=int)
    y[: n // 2] = 1
    x[: n // 2, FEATURE_NAMES.index("heart_rate_bpm")] += 100
    model = fit_model(x, {"sinus_tachycardia": y, "rbbb": 1 - y}, FEATURE_NAMES, min_positives=10)
    model.meta["suggest_classes"] = ["rbbb"]
    path = tmp_path / "toy.joblib"
    model.save(path)
    assert load_model(path) is not None
    assert load_model(tmp_path / "absent.joblib") is None

    cfg = load_app_config()
    cfg = cfg.model_copy(update={"ml": cfg.ml.model_copy(update={"model_path": str(path)})})
    result = run_pipeline(sig, mode="A", config=cfg)
    probs = result.measurements.ml_probabilities
    assert probs is not None and set(probs) == {"sinus_tachycardia", "rbbb"}
    assert probs["rbbb"] > AGREE_P                      # the toy model learned "rbbb" = normal HR
    assert SUGGEST_PREFIX + "rbbb" in result.findings.codes()
    assert any(a.startswith("[ML]") for a in result.audit_log)
    assert result.safety.alerts == []


@pytest.mark.parametrize("hr", [55.0, 75.0, 120.0])
def test_feature_vector_is_sized_and_mostly_defined(hr):
    sig = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10.0)
    m = run_pipeline(sig, mode="A").measurements
    x = features_from_measurements(m, sig.sampling_rate_hz)
    assert x.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(x).mean() > 0.8   # NaN allowed (HGB handles missing), but not many
