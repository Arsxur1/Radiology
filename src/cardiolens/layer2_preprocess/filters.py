"""Layer 2 — preprocessing filters (TZ §5).

Baseline-wander removal, configurable mains notch (50 Hz Uzbekistan / 60 Hz),
and an ST-preserving band-pass. Filters are zero-phase (``filtfilt``) to avoid
distorting the ST segment (TZ §5 "полосовой фильтр без искажения ST").
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

from cardiolens.contracts.signal import SignalRepresentation


def resample_signal(sig: SignalRepresentation, target_fs: float) -> SignalRepresentation:
    """Resample to ``target_fs`` (TZ §5). No-op if already at target."""
    if abs(sig.sampling_rate_hz - target_fs) < 1e-6:
        return sig
    from fractions import Fraction

    frac = Fraction(target_fs / sig.sampling_rate_hz).limit_denominator(1000)
    up, down = frac.numerator, frac.denominator
    new = resample_poly(sig.signal, up, down, axis=1).astype(np.float32)
    return sig.model_copy(update={"signal": new, "sampling_rate_hz": float(target_fs),
                                  "duration_s": float(new.shape[1] / target_fs)})


def _notch(row: np.ndarray, fs: float, freq: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=freq / (fs / 2.0), Q=q)
    return filtfilt(b, a, row)


def _bandpass(row: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = fs / 2.0
    high = min(high, nyq * 0.95)
    b, a = butter(N=3, Wn=[low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, row)


def preprocess(
    sig: SignalRepresentation,
    *,
    mains_frequency_hz: float,
    bandpass_low_hz: float,
    bandpass_high_hz: float,
    target_sampling_rate_hz: float | None = None,
) -> SignalRepresentation:
    """Full preprocessing chain (TZ §5): resample -> notch -> band-pass.

    Returns a new representation; the caller's signal is left untouched (it used
    to be overwritten, so re-running the pipeline on the same object filtered twice).
    """
    sig = sig.model_copy()
    if target_sampling_rate_hz is not None:
        sig = resample_signal(sig, target_sampling_rate_hz)
    fs = sig.sampling_rate_hz
    out = np.empty_like(sig.signal)
    for i in range(sig.n_leads):
        row = sig.signal[i].astype(np.float64)
        if mains_frequency_hz < fs / 2.0:
            row = _notch(row, fs, mains_frequency_hz)
        row = _bandpass(row, fs, bandpass_low_hz, bandpass_high_hz)
        out[i] = row.astype(np.float32)
    sig.signal = out
    return sig
