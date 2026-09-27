"""Feature vector for the learnable layer: Layer-3 measurements only (TZ §17).

Deliberately built from the same measurements the rules see (intervals, rhythm
statistics, per-lead ST, atrial activity), so the model stays explainable and
runs wherever the rules run. Missing measurements are NaN — the gradient-boosted
trees handle them natively.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.signal import STANDARD_12_LEADS

_SCALARS = [
    "heart_rate_bpm", "rr_cv", "rr_std_ms", "hrv_sdnn_ms", "p_wave_ratio", "p_onset_ratio",
    "pr_std_ms", "atrial_rate_bpm", "sokolow_lyon_uv", "qrs_amplitude_limb_uv", "n_beats",
]
_INTERVALS = ["pr_ms", "qrs_ms", "qt_ms", "qtc_bazett_ms", "qtc_fridericia_ms"]
_ST_FIELDS = ["j_point_uv", "j60_uv", "j80_uv", "qrs_net_uv"]

FEATURE_NAMES: list[str] = (
    _SCALARS + _INTERVALS
    + ["qrs_axis_deg", "rhythm_regular", "av_dissociation", "v1_terminal", "v6_terminal",
       "atrial_ratio", "n_peaked_t", "n_flat_t", "n_inverted_t", "n_u_wave",
       "rr_rmssd_ms", "rr_pnn50", "rr_sd1_ms", "rr_sd2_ms", "rr_range_ms"]
    + [f"{lead}_{f}" for lead in STANDARD_12_LEADS for f in _ST_FIELDS]
)


def _tri(v: bool | None) -> float:
    return np.nan if v is None else (1.0 if v else 0.0)


def _rr_stats(rpeaks: list[int], fs: float) -> list[float]:
    if len(rpeaks) < 4 or not fs:
        return [np.nan] * 5
    rr = np.diff(np.asarray(rpeaks, dtype=float)) / fs * 1000.0
    d = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(d * d)))
    pnn50 = float(np.mean(np.abs(d) > 50.0))
    sd1 = float(np.std(d) / np.sqrt(2.0))
    sd2 = float(np.sqrt(max(2.0 * np.var(rr) - 0.5 * np.var(d), 0.0)))
    return [rmssd, pnn50, sd1, sd2, float(rr.max() - rr.min())]


def features_from_measurements(m: Measurements, fs: float) -> np.ndarray:
    row: list[float] = []
    for name in _SCALARS:
        v = getattr(m, name, None)
        row.append(np.nan if v is None else float(v))
    for name in _INTERVALS:
        v = getattr(m.intervals, name, None)
        row.append(np.nan if v is None else float(v))
    axis = m.axes.qrs_deg
    row.append(np.nan if axis is None else float(axis))
    row.append(_tri(m.rhythm_regular))
    row.append(_tri(m.av_dissociation))
    row.append(_tri(m.qrs_v1_terminal_positive))
    row.append(_tri(m.qrs_v6_terminal_positive))
    row.append(np.nan if not (m.atrial_rate_bpm and m.heart_rate_bpm) else m.atrial_rate_bpm / m.heart_rate_bpm)
    mo = m.morphology
    row += [float(len(mo.peaked_t_leads)), float(len(mo.flat_t_leads)),
            float(len(mo.inverted_t_leads)), float(len(mo.u_wave_leads))]
    row += _rr_stats(m.r_peaks_samples, fs)
    st = {s.lead: s for s in mo.st}
    for lead in STANDARD_12_LEADS:
        s = st.get(lead)
        for f in _ST_FIELDS:
            v = getattr(s, f, None) if s is not None else None
            row.append(np.nan if v is None else float(v))
    out = np.asarray(row, dtype=np.float64)
    assert out.shape[0] == len(FEATURE_NAMES), (out.shape, len(FEATURE_NAMES))
    return out


def features_from_result(result) -> np.ndarray:
    """Features from a ``PipelineResult`` (measurements + the processed signal's rate)."""
    return features_from_measurements(result.measurements, result.signal.sampling_rate_hz)
