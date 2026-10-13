"""Suggestion eligibility (D33, D46): precise everywhere, never less sensitive than
the rules on any foreign set with enough positives — both transfer directions."""

from types import SimpleNamespace

from cardiolens.ml.train import suggest_classes


def _rep(**per_class):
    rules = {c: {"sens": v[0], "spec": v[1], "n": v[2]} for c, v in per_class.items()}
    model = {c: {"sens": v[3], "spec": v[4], "n": v[2]} for c, v in per_class.items()}
    return {"rules": rules, "model": model, "rules_or_model": model}


def _model(reports, transfer):
    return SimpleNamespace(classes=["rbbb", "lbbb"], meta={"reports": reports, "transfer": transfer})


def test_both_transfer_directions_must_hold():
    reports = {"holdout:a": _rep(rbbb=(0.7, 0.98, 70, 0.8, 0.99), lbbb=(0.7, 0.94, 80, 0.6, 0.99))}
    transfer = {
        "a->b": _rep(rbbb=(0.87, 0.96, 86, 0.93, 0.97), lbbb=(0.85, 0.87, 85, 0.83, 0.97)),
        "b->a": _rep(rbbb=(0.82, 0.98, 154, 0.75, 0.99), lbbb=(0.57, 0.94, 56, 0.48, 0.999)),
    }
    # rbbb wins a->b but loses b->a; lbbb loses both: nothing is autonomous.
    assert suggest_classes(_model(reports, transfer)) == []
    transfer["b->a"] = _rep(rbbb=(0.82, 0.98, 154, 0.85, 0.99), lbbb=(0.57, 0.94, 56, 0.48, 0.999))
    assert suggest_classes(_model(reports, transfer)) == ["rbbb"]


def test_small_foreign_sets_do_not_vote_but_specificity_always_does():
    reports = {
        "holdout:a": _rep(rbbb=(0.7, 0.98, 70, 0.8, 0.99), lbbb=(0.7, 0.94, 80, 0.75, 0.99)),
        "cross:kids": _rep(rbbb=(0.9, 0.99, 6, 0.2, 0.99), lbbb=(0.5, 0.99, 2, 0.5, 0.90)),
        "cross:big": _rep(rbbb=(0.8, 0.99, 60, 0.85, 0.99), lbbb=(0.5, 0.99, 40, 0.6, 0.99)),
    }
    # 6 positives cannot veto rbbb; lbbb's 0.90 specificity on the same set can.
    assert suggest_classes(_model(reports, {})) == ["rbbb"]


def test_a_class_without_a_large_foreign_set_is_not_eligible():
    reports = {
        "holdout:a": _rep(rbbb=(0.7, 0.98, 70, 0.8, 0.99), lbbb=(0.7, 0.94, 80, 0.75, 0.99)),
        "cross:kids": _rep(rbbb=(0.9, 0.99, 6, 0.9, 0.99), lbbb=(0.5, 0.99, 2, 0.5, 0.99)),
    }
    # Precise everywhere, but no foreign set with >= 20 positives ever compared it to the rules.
    assert suggest_classes(_model(reports, {})) == []


def test_no_reports_means_no_suggestions():
    assert suggest_classes(_model({}, {})) == []
