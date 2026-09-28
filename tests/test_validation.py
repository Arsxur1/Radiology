"""Validation harness + metrics tests (TZ §16)."""

from cardiolens.validation import (
    ECG_ARRHYTHMIA_SCORED_LABELS,
    PTBXL_SCORED_LABELS,
    PTBXL_SCP_TO_CODE,
    SNOMED_TO_CODE,
    IntervalAccuracy,
    build_synthetic_dataset,
    gold_intervals_ms,
    run_validation,
)
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


def test_real_dataset_scored_labels_derive_from_maps():
    # The scored-label set for a real dataset must be exactly the distinct
    # finding codes its label map can supply — never an always-zero row for a
    # code the source can't produce, never a mappable code left unscored.
    assert PTBXL_SCORED_LABELS == sorted(set(PTBXL_SCP_TO_CODE.values()))
    assert ECG_ARRHYTHMIA_SCORED_LABELS == sorted(set(SNOMED_TO_CODE.values()))
    # Codes the pipeline reports (layer4) so scoring is against real predictions.
    assert {"rbbb", "lbbb", "atrial_fibrillation", "lvh"} <= set(PTBXL_SCORED_LABELS)
    # Flutter is scored on PTB-XL too (AFLT), so a weak detector stays visible.
    assert PTBXL_SCP_TO_CODE["AFLT"] == "atrial_flutter"


def test_ptbxl_st_mapping_uses_st_codes_not_mi_scars():
    # ST elevation gold must come from the SCP ST-elevation statement, not the
    # MI-superclass codes (usually chronic Q-wave scars without a current injury
    # current) — mapping those to acute ST elevation mislabels the benchmark.
    assert PTBXL_SCP_TO_CODE.get("STE_") == "st_elevation"
    assert PTBXL_SCP_TO_CODE.get("STD_") == "st_depression"
    assert "AMI" not in PTBXL_SCP_TO_CODE and "IMI" not in PTBXL_SCP_TO_CODE


def test_explicit_labels_override_is_scored():
    # run_validation must score exactly the labels passed, in support order.
    report = run_validation(build_synthetic_dataset(), labels=["st_elevation"])
    assert set(report.per_class) == {"st_elevation"}
    assert report.per_class["st_elevation"].sensitivity == 1.0


def test_gold_intervals_from_ludb_triplets():
    # LUDB annotations: onset "(" / peak (p|N|t) / offset ")" triplets, ordered
    # P, QRS, T per beat. At 1 kHz, one sample = 1 ms.
    symbols = ["(", "p", ")", "(", "N", ")", "(", "t", ")"]
    samples = [100, 120, 140, 300, 320, 400, 500, 560, 640]
    pr, qrs, qt = gold_intervals_ms(symbols, samples, fs=1000.0)
    assert pr == 200.0   # P onset 100 -> QRS onset 300
    assert qrs == 100.0  # QRS onset 300 -> offset 400
    assert qt == 340.0   # QRS onset 300 -> T offset 640


def test_interval_accuracy_bias_and_mae():
    acc = IntervalAccuracy("QRS")
    acc.add(132, 95)     # +37
    acc.add(120, 100)    # +20
    acc.add(None, 90)    # skipped
    assert acc.n == 2
    assert acc.bias_ms == 28.5   # median signed error
    assert acc.mae_ms == 28.5


def test_select_ptbxl_rows_filters_official_fold():
    import pandas as pd

    from cardiolens.validation.harness import select_ptbxl_rows

    db = pd.DataFrame({"strat_fold": [1, 9, 10, 10, 10, 3]},
                      index=pd.Index([1, 2, 3, 4, 5, 6], name="ecg_id"))
    # Fold 10 is PTB-XL's recommended test split.
    assert list(select_ptbxl_rows(db, folds=[10]).index) == [3, 4, 5]
    # The limit applies after the fold filter, not before it.
    assert list(select_ptbxl_rows(db, folds=[10], limit=2).index) == [3, 4]
    assert list(select_ptbxl_rows(db, folds=[9, 10]).index) == [2, 3, 4, 5]
    # No filter keeps the previous behaviour (first N rows).
    assert list(select_ptbxl_rows(db, limit=2).index) == [1, 2]


def test_challenge2021_snomed_codes_map_to_matching_findings():
    # Challenge 2021 scored classes that match emitted findings.
    assert SNOMED_TO_CODE["10370003"] == "paced_rhythm"
    assert SNOMED_TO_CODE["59931005"] == "t_inversion"
    assert SNOMED_TO_CODE["427172004"] == SNOMED_TO_CODE["17338001"] == "ventricular_ectopy"
    # Ambiguous codes stay unmapped rather than being equated with a narrower finding.
    assert "426627000" not in SNOMED_TO_CODE   # generic bradycardia
    assert "164947007" not in SNOMED_TO_CODE   # prolonged PR (scored separately from IAVB)


def test_ectopy_subtypes_satisfy_pvc_gold_label():
    """A 'PVC' gold label is detected when the pipeline reports couplet/bigeminy/VT."""
    from cardiolens.validation.harness import expand_implied

    for sub in ("ventricular_couplet", "ventricular_bigeminy", "ventricular_tachycardia"):
        assert "ventricular_ectopy" in expand_implied({sub})
    assert expand_implied({"sinus_rhythm"}) == {"sinus_rhythm"}
