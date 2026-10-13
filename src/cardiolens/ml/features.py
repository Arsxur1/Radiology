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

SUMMARY_FEATURE_NAMES: list[str] = (
    _SCALARS + _INTERVALS
    + ["qrs_axis_deg", "rhythm_regular", "av_dissociation", "v1_terminal", "v6_terminal",
       "atrial_ratio", "n_peaked_t", "n_flat_t", "n_inverted_t", "n_u_wave",
       "rr_rmssd_ms", "rr_pnn50", "rr_sd1_ms", "rr_sd2_ms", "rr_range_ms"]
    + [f"{lead}_{f}" for lead in STANDARD_12_LEADS for f in _ST_FIELDS]
)

# v2 (open item 8): the average beat's shape per lead. The summary features carry
# amplitudes and intervals but not the waveform; the model transferred poorly across
# databases without it. Window [-BEAT_PRE_MS, +BEAT_POST_MS] around R, PR-segment
# baseline, amplitude-normalised by the lead's QRS peak, BEAT_POINTS samples, plus
# the QRS/T energy ratio and the signed QRS/T peak ratio per lead.
FEATURE_SET = "v2"
BEAT_PRE_MS, BEAT_POST_MS, BEAT_POINTS = 250.0, 450.0, 24
SHAPE_FEATURE_NAMES: list[str] = (
    [f"{lead}_beat{k:02d}" for lead in STANDARD_12_LEADS for k in range(BEAT_POINTS)]
    + [f"{lead}_{f}" for lead in STANDARD_12_LEADS for f in ("qrs_t_energy_ratio", "t_qrs_peak_ratio")]
)
FEATURE_NAMES: list[str] = SUMMARY_FEATURE_NAMES + SHAPE_FEATURE_NAMES


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


def beat_shape_features(signal: np.ndarray, lead_names: list[str], rpeaks: list[int], fs: float) -> np.ndarray:
    """Per-lead median beat (normalised, BEAT_POINTS samples) + energy/peak ratios."""
    n_lead = len(STANDARD_12_LEADS)
    out = np.full(n_lead * BEAT_POINTS + 2 * n_lead, np.nan)
    rp = np.asarray(rpeaks, dtype=int)
    if rp.size < 2 or not fs:
        return out
    pre, post = int(BEAT_PRE_MS / 1000.0 * fs), int(BEAT_POST_MS / 1000.0 * fs)
    b_lo, b_hi = int(0.10 * fs), int(0.05 * fs)     # PR-segment baseline window before R
    q_w, t_lo, t_hi = int(0.06 * fs), int(0.12 * fs), int(0.40 * fs)
    for li, lead in enumerate(STANDARD_12_LEADS):
        if lead not in lead_names:
            continue
        row = np.asarray(signal[lead_names.index(lead)], dtype=np.float64)
        beats = []
        for r in rp:
            if r - pre < 0 or r + post >= len(row):
                continue
            base = float(np.median(row[r - b_lo: r - b_hi])) if r - b_lo >= 0 else float(np.median(row[max(0, r - pre): r]))
            beats.append(row[r - pre: r + post] - base)
        if len(beats) < 2:
            continue
        med = np.median(np.asarray(beats), axis=0)
        qrs = med[pre - q_w: pre + q_w]
        scale = float(np.max(np.abs(qrs))) if qrs.size else 0.0
        if scale <= 0:
            continue
        idx = np.linspace(0, len(med) - 1, BEAT_POINTS)
        out[li * BEAT_POINTS: (li + 1) * BEAT_POINTS] = np.interp(idx, np.arange(len(med)), med) / scale
        t_seg = med[pre + t_lo: pre + t_hi]
        e_qrs, e_t = float(np.sum(qrs ** 2)), float(np.sum(t_seg ** 2))
        out[n_lead * BEAT_POINTS + 2 * li] = e_qrs / (e_t + 1e-9)
        t_peak = t_seg[np.argmax(np.abs(t_seg))] if t_seg.size else 0.0
        out[n_lead * BEAT_POINTS + 2 * li + 1] = float(t_peak) / scale
    return out


def features_from_measurements(m: Measurements, fs: float, signal=None, lead_names=None) -> np.ndarray:
    """Summary features (+ beat-shape features when the processed signal is given;
    NaN otherwise, which the trees treat as missing)."""
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
    summary = np.asarray(row, dtype=np.float64)
    if signal is not None and lead_names is not None:
        shape = beat_shape_features(signal, list(lead_names), m.r_peaks_samples, fs)
    else:
        shape = np.full(len(SHAPE_FEATURE_NAMES), np.nan)
    out = np.concatenate([summary, shape])
    assert out.shape[0] == len(FEATURE_NAMES), (out.shape, len(FEATURE_NAMES))
    return out


def features_from_result(result) -> np.ndarray:
    """Features from a ``PipelineResult`` (measurements + the processed signal)."""
    sig = result.signal
    return features_from_measurements(result.measurements, sig.sampling_rate_hz, sig.signal, sig.lead_names)
