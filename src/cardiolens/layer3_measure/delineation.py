"""Robust delineation helpers layered over neurokit's dwt output (TZ §6).

Real 12-lead recordings (PhysioNet ecg-arrhythmia) exposed two weaknesses of the
pure dwt delineation: QRS width is under-estimated (bundle branch blocks read as
narrow) and the P onset lands late (PR under-estimated). These helpers derive QRS
bounds from the multi-lead spatial-velocity envelope and refine P onsets by
amplitude. Where a beat cannot be delineated the value is left NaN — never
fabricated (TZ §0).
"""

from __future__ import annotations

import numpy as np

# Envelope threshold as a fraction of the per-beat envelope maximum. Calibrated on
# PhysioNet ecg-arrhythmia: at 0.12 the slurred terminal forces of a bundle branch
# block fall below threshold (RBBB read as 62-74 ms); at 0.05 RBBB/LBBB measure
# 112-190 ms while normal QRS stays <= 102 ms.
QRS_ENVELOPE_FRACTION = 0.05
# A width beyond this is not a QRS (envelope ran into the T wave) -> NaN.
QRS_MAX_MS = 220.0
# Search half-window around each R peak, and how long the envelope must stay
# below threshold to count as "outside the QRS".
QRS_SEARCH_MS = 150.0
QRS_QUIET_MS = 8.0

# P-onset refinement: search window before QRS onset, onset criterion and cap.
P_SEARCH_BEFORE_MS = 320.0
P_SEARCH_UNTIL_MS = 40.0
# Onset = where the smoothed deflection drops below this fraction of the P peak
# (with an absolute floor above the noise of a 0.5-40 Hz band-passed ECG), at most
# P_ONSET_MAX_MS before the peak; if never met, the onset is the start of the
# supra-threshold run (NOT a fixed offset, which inflated PR by 60-80 ms).
P_ONSET_FRACTION = 0.15
P_ONSET_MIN_LEVEL_UV = 12.0
P_ONSET_MAX_MS = 80.0
P_SMOOTH_MS = 10.0
P_BASELINE_MS = 30.0
# A P-wave candidate run above the floor must be this wide (ms): narrower is
# noise, wider is a T wave.
P_RUN_WIDTH_MS = (20.0, 160.0)
# Amplitude floor for P candidates as a fraction of the R reference (lower than
# the AF gate: we only need a plausible P, not a certainly organised one).
P_ONSET_MIN_FRACTION = 0.05
P_ONSET_MIN_UV = 30.0

# Terminal-QRS window (last N ms before the QRS offset) for BBB typing.
TERMINAL_MS = 40.0


def spatial_velocity(signal: np.ndarray, fs: float) -> np.ndarray:
    """RMS across leads of the first derivative, smoothed over ~10 ms.

    The spatial-velocity envelope peaks sharply during the QRS in every lead at
    once, which makes the QRS bounds far less sensitive to single-lead morphology
    (notches, R', deep S) than a one-lead delineation.
    """
    x = signal.astype(np.float64)
    d = np.diff(x, axis=1, prepend=x[:, :1])
    env = np.sqrt(np.mean(d * d, axis=0))
    k = max(1, int(round(0.010 * fs)))
    return np.convolve(env, np.ones(k) / k, mode="same")


def qrs_bounds_envelope(
    signal: np.ndarray, rpeaks: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-beat QRS onset/offset (sample indices, NaN when not found)."""
    n_beats = len(rpeaks)
    on = np.full(n_beats, np.nan)
    off = np.full(n_beats, np.nan)
    if n_beats == 0:
        return on, off
    env = spatial_velocity(signal, fs)
    n = env.size
    half = int(QRS_SEARCH_MS / 1000.0 * fs)
    quiet = max(1, int(round(QRS_QUIET_MS / 1000.0 * fs)))
    for i, r in enumerate(rpeaks):
        a, b = max(0, int(r) - half), min(n, int(r) + half + 1)
        if b - a < 3:
            continue
        seg = env[a:b]
        thr = QRS_ENVELOPE_FRACTION * float(seg.max())
        below = seg < thr
        ri = int(r) - a
        # Onset: walk left from R until the envelope has been quiet for `quiet`.
        run, j = 0, ri
        while j > 0:
            j -= 1
            run = run + 1 if below[j] else 0
            if run >= quiet:
                on[i] = a + j + quiet
                break
        # Offset: walk right from R likewise.
        run, j = 0, ri
        while j < len(seg) - 1:
            j += 1
            run = run + 1 if below[j] else 0
            if run >= quiet:
                off[i] = a + j - quiet + 1
                break
        if np.isfinite(on[i]) and np.isfinite(off[i]) and (off[i] - on[i]) > QRS_MAX_MS / 1000.0 * fs:
            on[i] = off[i] = np.nan
    return on, off


def refine_p_onsets(
    row: np.ndarray,
    qrs_on: np.ndarray,
    t_off: np.ndarray | None,
    fs: float,
    min_amp_uv: float,
) -> np.ndarray:
    """Amplitude-based P onset per beat on one lead (NaN when no P is found).

    Searches [qrs_on − 320 ms, qrs_on − 40 ms] (not before the previous T offset)
    with the PR-segment median as baseline. Candidate deflections are contiguous
    runs above ``min_amp_uv`` that are 20–160 ms wide (a T wave is wider, noise is
    narrower); the run **closest to the QRS** is the P wave — taking the largest
    run instead picks up the previous T/U wave in bradycardia. The onset is where
    the deflection drops below 10 % of the P amplitude walking back from its peak.
    """
    n_beats = len(qrs_on)
    p_on = np.full(n_beats, np.nan)
    before = int(P_SEARCH_BEFORE_MS / 1000.0 * fs)
    until = int(P_SEARCH_UNTIL_MS / 1000.0 * fs)
    base_w = int(P_BASELINE_MS / 1000.0 * fs)
    need = max(1, int(round(0.006 * fs)))
    cap = int(P_ONSET_MAX_MS / 1000.0 * fs)
    w_min, w_max = int(P_RUN_WIDTH_MS[0] / 1000.0 * fs), int(P_RUN_WIDTH_MS[1] / 1000.0 * fs)
    smooth = max(1, int(round(P_SMOOTH_MS / 1000.0 * fs)))
    for i in range(n_beats):
        if not np.isfinite(qrs_on[i]):
            continue
        on = int(qrs_on[i])
        lo, hi = on - before, on - until
        if t_off is not None and i > 0 and i - 1 < len(t_off) and np.isfinite(t_off[i - 1]):
            lo = max(lo, int(t_off[i - 1]) + int(0.02 * fs))
        lo = max(lo, 0)
        if hi - lo < int(0.04 * fs) or on - base_w < 0:
            continue
        base = float(np.median(row[on - base_w:on]))
        seg = row[lo:hi] - base
        mask = np.abs(seg) >= min_amp_uv
        if not mask.any():
            continue
        # Contiguous runs above the floor, scanned from the QRS backwards.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(int), [0]))))
        runs = list(zip(edges[::2], edges[1::2]))
        k = None
        for start, stop in reversed(runs):
            if w_min <= stop - start <= w_max:
                k = start + int(np.argmax(np.abs(seg[start:stop])))
                break
        # `start` of the chosen run doubles as the onset fallback below.
        if k is None:
            continue
        amp = float(abs(seg[k]))
        sm = np.convolve(np.abs(seg), np.ones(smooth) / smooth, mode="same")
        level = max(P_ONSET_FRACTION * amp, P_ONSET_MIN_LEVEL_UV)
        limit = max(0, k - cap)
        onset, run, j = max(limit, start), 0, k
        while j > limit:
            j -= 1
            if sm[j] < level:
                run += 1
                if run >= need:
                    onset = j + run
                    break
            else:
                run = 0
        p_on[i] = lo + onset
    return p_on


def terminal_qrs_sign(
    row: np.ndarray, rpeaks: np.ndarray, qrs_off: np.ndarray, fs: float
) -> bool | None:
    """Sign of the terminal QRS forces on one lead (True = positive).

    Averages the last ``TERMINAL_MS`` before the QRS offset; when the offset is
    unknown for a beat, falls back to the 20–60 ms window after R.
    """
    if len(rpeaks) == 0:
        return None
    baseline = float(np.median(row))
    w = int(TERMINAL_MS / 1000.0 * fs)
    vals = []
    for i, r in enumerate(rpeaks):
        if i < len(qrs_off) and np.isfinite(qrs_off[i]):
            b = int(qrs_off[i])
            a = b - w
        else:
            a, b = int(r) + int(0.02 * fs), int(r) + int(0.06 * fs)
        if 0 <= a < b <= len(row):
            vals.append(float(np.mean(row[a:b]) - baseline))
    if not vals:
        return None
    return bool(np.median(vals) > 0)
