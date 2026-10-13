"""Validation harness + metrics tests (TZ §16)."""

from cardiolens.validation import build_synthetic_dataset, run_validation
from cardiolens.validation.metrics import (
    ClassMetrics,
    IntervalError,
    macro_recall,
    score_findings,
)


def test_class_metrics_math():
    m = ClassMetrics("x", tp=8, fp=2, tn=90, fn=2)
    assert m.sensitivity == 0.8
    assert m.specificity == round(90 / 92, 3)
    assert m.support == 10


def test_score_findings_confusion():
    preds = [{"a"}, {"a", "b"}, set()]
    gold = [{"a"}, {"b"}, {"a"}]
    labels = ["a", "b"]
    m = score_findings(preds, gold, labels)
    assert m["a"].tp == 1 and m["a"].fp == 1 and m["a"].fn == 1
    assert m["b"].tp == 1


def test_interval_error_mae():
    e = IntervalError("PR")
    e.add(160, 150)
    e.add(140, 150)
    e.add(None, 150)  # skipped
    assert e.n == 2 and e.mae == 10.0


def test_synthetic_validation_runs_with_sane_metrics():
    report = run_validation(build_synthetic_dataset())
    assert report.n_cases >= 15
    # Detectors should recall the majority of clear positives.
    assert report.macro_sensitivity is not None and report.macro_sensitivity >= 0.7
    # Interval errors are bounded (delineation is imperfect but not wild).
    assert report.pr_mae_ms is not None and report.pr_mae_ms < 80
    assert report.qt_mae_ms is not None and report.qt_mae_ms < 80
    # High-stakes classes detected.
    assert report.per_class["st_elevation"].sensitivity == 1.0
    assert report.per_class["sinus_tachycardia"].sensitivity == 1.0


def test_macro_recall_none_when_no_support():
    assert macro_recall({"x": ClassMetrics("x")}) is None
