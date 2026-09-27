"""Multi-lead tangent-method T offset (TZ §6): QT no longer depends on one lead's dwt mark."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer3_measure.delineation import t_offset_tangent
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline


def _run(**kw):
    return run_pipeline(generate_synthetic_ecg(**kw), mode="A", provider=StubProvider())


@pytest.mark.parametrize("hr,qt", [(60, 400), (75, 360), (90, 330), (50, 440)])
def test_qt_tracks_stimulus_with_tangent_method(hr, qt):
    m = _run(heart_rate_bpm=hr, qt_ms=qt, seed=3).measurements
    assert m.intervals.qt_source == "tangent"
    assert m.intervals.qt_lead is not None
    assert m.intervals.qt_ms == pytest.approx(qt, abs=25)


def test_inverted_t_is_still_delineated():
    m = _run(heart_rate_bpm=70, qt_ms=380, t_amp_uv=-300, seed=4).measurements
    assert m.intervals.qt_source == "tangent"
    assert m.intervals.qt_ms == pytest.approx(380, abs=30)


def test_tangent_offset_on_a_clean_t_wave():
    fs = 500.0
    t = np.arange(0, 1.0, 1 / fs)
    # QRS-free beat: flat baseline + one Gaussian T centred at 0.40 s (sigma 40 ms).
    x = 300.0 * np.exp(-0.5 * ((t - 0.40) / 0.04) ** 2)
    off = t_offset_tangent(x, qrs_on=0.10 * fs, qrs_off=0.15 * fs, rr_s=1.0, fs=fs)
    assert off is not None
    # Tangent at the steepest descent (peak + sigma) meets baseline ~ peak + 2.4 sigma.
    assert 0.47 * fs <= off <= 0.52 * fs


def test_small_t_waves_are_rejected():
    fs = 500.0
    t = np.arange(0, 1.0, 1 / fs)
    x = 40.0 * np.exp(-0.5 * ((t - 0.40) / 0.04) ** 2)
    assert t_offset_tangent(x, 0.10 * fs, 0.15 * fs, 1.0, fs) is None


def test_long_qt_stimulus_is_flagged():
    r = _run(heart_rate_bpm=60, qt_ms=520, seed=5)
    assert "long_qt" in r.findings.codes()
    assert r.measurements.intervals.qtc_bazett_ms > 480
