"""Oversense rejection on a 3x4 page compares beats within their own lead set (D52).

A digitised page carries each panel lead for 2.5 s only; the other channels are
zero there. The spatial-velocity envelope of a beat under aVR/aVL/aVF is then a
fraction of one under V1–V3, and a whole-record reference rejected an entire
column of true beats (HR fell by a quarter, RR variability tripled, false AF).
"""

import numpy as np

from cardiolens.layer1_ingestion.digitize import LAYOUT_3X4, PANEL_SECONDS, RHYTHM_LEAD
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg
from cardiolens.layer3_measure.measure import _reject_non_qrs_detections


def _page_like(scale_by_column=(0.3, 0.3, 3.0, 3.0)):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, noise_uv=0.0)
    fs = sig.sampling_rate_hz
    seg = int(PANEL_SECONDS * fs)
    windows = {}
    for col, leads in enumerate(zip(*LAYOUT_3X4)):
        for lead in leads:
            li = sig.lead_names.index(lead)
            if lead == RHYTHM_LEAD:
                windows[lead] = (0.0, 4 * PANEL_SECONDS)
                continue
            row = sig.signal[li].copy()
            sig.signal[li] = 0.0
            sig.signal[li, col * seg:(col + 1) * seg] = row[col * seg:(col + 1) * seg] * scale_by_column[col]
            windows[lead] = (col * PANEL_SECONDS, (col + 1) * PANEL_SECONDS)
    return sig, windows


def _true_rpeaks(sig):
    import neurokit2 as nk

    _, info = nk.ecg_peaks(sig.lead("II").astype(float), sampling_rate=sig.sampling_rate_hz)
    return np.asarray(info["ECG_R_Peaks"], dtype=int)


def test_whole_record_reference_drops_a_limb_column_but_windows_keep_it():
    sig, windows = _page_like()
    rpeaks = _true_rpeaks(sig)
    fs = sig.sampling_rate_hz
    assert len(rpeaks) >= 10
    without = _reject_non_qrs_detections(sig.signal, rpeaks, fs)
    with_windows = _reject_non_qrs_detections(sig.signal, rpeaks, fs, windows=windows, lead_names=sig.lead_names)
    assert len(without) < len(rpeaks)              # the failure mode this guards against
    assert len(with_windows) == len(rpeaks)        # every true beat survives


def test_t_wave_oversense_is_still_rejected_with_windows():
    sig, windows = _page_like()
    rpeaks = _true_rpeaks(sig)
    fs = sig.sampling_rate_hz
    fake = rpeaks[:-1] + int(0.30 * fs)            # one detection on every T wave
    mixed = np.sort(np.concatenate([rpeaks, fake]))
    kept = _reject_non_qrs_detections(sig.signal, mixed, fs, windows=windows, lead_names=sig.lead_names)
    assert set(kept.tolist()) == set(rpeaks.tolist())


def test_digital_record_unchanged_without_windows():
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    rpeaks = _true_rpeaks(sig)
    fs = sig.sampling_rate_hz
    a = _reject_non_qrs_detections(sig.signal, rpeaks, fs)
    b = _reject_non_qrs_detections(sig.signal, rpeaks, fs, windows=None, lead_names=sig.lead_names)
    assert np.array_equal(a, b) and len(a) == len(rpeaks)


def test_column_beat_detector_finds_beats_the_rhythm_strip_loses():
    """D55: a near-isoelectric lead II on the page loses beats to the single-lead
    detector; the column detector sees the chest leads of each column."""
    from cardiolens.layer3_measure.measure import _page_beats

    sig, windows = _page_like(scale_by_column=(1.0, 1.0, 1.0, 1.0))
    true = _true_rpeaks(sig)
    fs = sig.sampling_rate_hz
    ii = sig.lead_names.index("II")
    sig.signal[ii] *= 0.12                                # rhythm strip almost flat
    found = _page_beats(sig, windows, sig.signal[ii].astype(float), fs)
    assert found is not None
    beats, heights = found
    assert len(beats) >= len(true) and np.all(heights >= 0.35)
    for r in true:                                          # every record beat within 60 ms of one
        assert np.min(np.abs(beats - r)) <= 0.06 * fs


def test_column_beat_detector_is_silent_without_windows():
    from cardiolens.layer3_measure.measure import _page_beats

    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    assert _page_beats(sig, {}, sig.lead("II").astype(float), sig.sampling_rate_hz) is None
