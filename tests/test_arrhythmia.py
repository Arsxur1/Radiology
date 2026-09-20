"""Beat-classifier / ventricular-ectopy tests (TZ §7.1).

CI-safe: no external MIT-BIH data. Validates the runtime contract (feature
extraction, no-op without a model, the committed model loads and behaves) and
the finding thresholds. Classifier accuracy is validated offline on MIT-BIH
(DS1->DS2, VEB ~0.82 sens / 0.97 PPV; see DECISIONS.md).
"""

import numpy as np

from cardiolens.arrhythmia import (
    BeatClassifier,
    EctopySummary,
    ectopy_findings,
    extract_beat_features,
    summarize_ectopy,
)
from cardiolens.arrhythmia.beats import FEATURES
from cardiolens.config import configs_dir
from cardiolens.layer1_ingestion import generate_synthetic_ecg


def test_ectopy_findings_thresholds():
    # <2 ventricular beats -> no finding (single beat within per-beat error).
    assert ectopy_findings(EctopySummary(n_beats=12, n_ventricular=1)) == []
    fs = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=4))
    assert [f.code for f in fs] == ["ventricular_ectopy"]
    big = ectopy_findings(EctopySummary(n_beats=12, n_ventricular=6, bigeminy=True))
    assert [f.code for f in big] == ["ventricular_bigeminy"]


def test_unfitted_classifier_is_noop():
    clf = BeatClassifier.load(configs_dir() / "does_not_exist.json")
    assert clf.fitted is False
    assert clf.predict(np.zeros((3, len(FEATURES)))) == ["N", "N", "N"]


def test_committed_model_loads_and_predicts():
    clf = BeatClassifier.load(configs_dir() / "beat_lda.json")
    assert clf.fitted is True
    assert set(clf.classes) == {"N", "S", "V"}
    # Random feature rows classify to valid labels without error.
    rng = np.random.default_rng(0)
    preds = clf.predict(rng.normal(size=(5, len(clf.mu))))
    assert all(p in clf.classes for p in preds)


def test_no_false_ectopy_on_normal_synthetic():
    # A clean sinus rhythm must not raise ventricular ectopy.
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    import neurokit2 as nk
    lead = sig.lead("II")
    _, info = nk.ecg_peaks(lead, sampling_rate=sig.sampling_rate_hz)
    rpeaks = info["ECG_R_Peaks"]
    clf = BeatClassifier.load(configs_dir() / "beat_lda.json")
    summary = summarize_ectopy(lead, rpeaks, sig.sampling_rate_hz, clf)
    assert summary.n_beats > 0
    assert summary.burden <= 0.15  # essentially no ventricular beats on clean sinus


def test_feature_matrix_shape():
    sig = generate_synthetic_ecg(heart_rate_bpm=75)
    import neurokit2 as nk
    lead = sig.lead("II")
    _, info = nk.ecg_peaks(lead, sampling_rate=sig.sampling_rate_hz)
    X, idx = extract_beat_features(lead, info["ECG_R_Peaks"], sig.sampling_rate_hz)
    assert X.shape[0] == len(idx)
    assert X.shape[1] == 8
