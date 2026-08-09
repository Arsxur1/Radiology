"""Physician feedback capture tests (TZ §13)."""

from cardiolens.feedback import Feedback, FeedbackStore, FindingFeedback


def _fb(analysis_id="abc"):
    return Feedback(
        analysis_id=analysis_id, reviewer="dr. X", mode="A",
        findings=[
            FindingFeedback(code="st_elevation", verdict="confirmed", confidence=0.8),
            FindingFeedback(code="lvh", verdict="rejected", confidence=0.6),
            FindingFeedback(code="t_inversion", verdict="corrected",
                            confidence=0.5, corrected_code="juvenile_t_pattern"),
        ],
        overall_agreement=False,
    )


def test_append_and_read(tmp_path):
    store = FeedbackStore.at(tmp_path / "fb.jsonl")
    store.append(_fb("a1"))
    store.append(_fb("a2"))
    all_fb = store.all()
    assert len(all_fb) == 2
    assert {f.analysis_id for f in all_fb} == {"a1", "a2"}


def test_calibration_samples_maps_verdicts():
    samples = _fb().calibration_samples()
    # confirmed -> 1, rejected -> 0, corrected -> skipped
    assert (0.8, 1) in samples
    assert (0.6, 0) in samples
    assert len(samples) == 2


def test_calibration_dataset_aggregates(tmp_path):
    store = FeedbackStore.at(tmp_path / "fb.jsonl")
    store.append(_fb("a1"))
    store.append(_fb("a2"))
    assert len(store.calibration_dataset()) == 4


def test_missing_log_is_empty(tmp_path):
    assert FeedbackStore.at(tmp_path / "none.jsonl").all() == []
