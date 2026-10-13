"""Trace tracking on a 3x4 page (D47): own trace followed across the row boundary,
foreign ink rejected, QRS width and R amplitude preserved."""

import numpy as np

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import (
    LAYOUT_3X4,
    PANEL_SECONDS,
    digitize_page,
    digitize_page_to_signal,
    render_ecg_page,
)
from cardiolens.pipeline import run_pipeline


def _corr(a, b):
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1])


def _page(tmp_path, **kw):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=10.0, seed=5, **kw)
    path = str(tmp_path / "page.png")
    render_ecg_page(sig, path, calibration_pulse=True)
    return sig, path


def test_tall_complexes_crossing_into_the_next_row_are_followed(tmp_path):
    # 2x amplitude: R/S excursions reach the neighbouring rows of the page. The
    # per-column median used to mix the intruding trace into both leads; tracking
    # keeps each panel on its own trace.
    sig, path = _page(tmp_path, amplitude_scale=2.0)
    seg = int(PANEL_SECONDS * sig.sampling_rate_hz)
    tracked = digitize_page(path)
    median = digitize_page(path, tracked=False)
    r_tracked = [_corr(sig.lead(lead)[c * seg:(c + 1) * seg], tracked[lead])
                 for row in LAYOUT_3X4 for c, lead in enumerate(row)]
    r_median = [_corr(sig.lead(lead)[c * seg:(c + 1) * seg], median[lead])
                for row in LAYOUT_3X4 for c, lead in enumerate(row)]
    assert min(r_tracked) > 0.85 and np.mean(r_tracked) > 0.9
    assert np.mean(r_tracked) >= np.mean(r_median) - 0.01
    assert _corr(sig.lead("II"), tracked["II_rhythm"]) > 0.9


def test_qrs_width_and_r_amplitude_survive_digitisation(tmp_path):
    sig, path = _page(tmp_path)
    ref = run_pipeline(sig, mode="A").measurements
    dig = run_pipeline(digitize_page_to_signal(path), mode="A").measurements
    # Before D47 the digitised QRS came out 24 ms (median) wider than the record.
    assert abs(dig.intervals.qrs_ms - ref.intervals.qrs_ms) <= 10
    assert abs(dig.heart_rate_bpm - ref.heart_rate_bpm) < 2
    # The per-column median halved the R amplitude on narrow complexes.
    assert dig.qrs_amplitude_limb_uv > 0.8 * ref.qrs_amplitude_limb_uv


def test_untracked_mode_is_still_available(tmp_path):
    _sig, path = _page(tmp_path)
    traces = digitize_page(path, tracked=False)
    assert set(traces) == {lead for row in LAYOUT_3X4 for lead in row} | {"II_rhythm"}
