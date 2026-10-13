"""Streaming / online processing tests (TZ §4)."""

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.streaming import SlidingWindower, StreamProcessor


def test_windower_emits_overlapping_windows():
    w = SlidingWindower(2, 100, window_s=10, hop_s=5)  # window 1000, hop 500
    starts = []
    for _ in range(30):  # push 30 one-second chunks
        for start, win in w.push(np.zeros((2, 100), dtype="float32")):
            assert win.shape == (2, 1000)
            starts.append(start)
    # Windows at absolute sample 0, 500, 1000, 1500, 2000 (0,5,10,15,20 s).
    assert starts == [0, 500, 1000, 1500, 2000]


def test_windower_no_partial_window():
    w = SlidingWindower(1, 100, window_s=10, hop_s=5)
    # Only 8 s pushed -> no complete 10 s window yet.
    out = w.push(np.zeros((1, 800), dtype="float32"))
    assert out == []


def test_windower_bad_chunk_shape():
    w = SlidingWindower(3, 100)
    try:
        w.push(np.zeros((2, 100)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for wrong lead count")


def test_windower_invalid_hop():
    try:
        SlidingWindower(1, 100, window_s=5, hop_s=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError when hop > window")


def test_stream_detects_new_critical_at_transition():
    # 20 s normal, then 20 s inferior STEMI.
    normal = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=20)
    stemi = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=20, seed=99,
                                   st_elevation_uv=250, st_leads=["II", "III", "aVF"])
    rec = SignalRepresentation(
        signal=np.concatenate([normal.signal, stemi.signal], axis=1),
        sampling_rate_hz=500, lead_names=normal.lead_names, duration_s=40,
        source="synthetic",
    )
    sp = StreamProcessor(rec.lead_names, 500, window_s=10, hop_s=5, mode="B")
    results = sp.process_signal(rec, chunk_s=2.0)

    assert len(results) == 7
    # No critical while the recording is still normal.
    for wr in results[:3]:
        assert "st_elevation" not in wr.critical_codes
        assert wr.result.report.mode == "B"
    # Once STEMI develops, st_elevation appears exactly once as "new critical".
    new_st = [wr for wr in results if "st_elevation" in wr.new_critical]
    assert len(new_st) == 1
    assert new_st[0].t_start_s >= 15.0
    assert new_st[0].result.report.mode == "escalate"
