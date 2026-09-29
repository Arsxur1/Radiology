"""QRS voltages for the hypertrophy criteria (TZ §7.4, open item 2).

Sokolow-Lyon alone reached 0.35 sensitivity on PTB-XL's LVH label; Cornell and
Romhilt-Estes need R/S amplitudes in aVL, V1-V3, V5-V6 and the limb leads, the
intrinsicoid deflection time in V5/V6 and the terminal P force in V1. Every value is
a median over the beats inside the lead's data window.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.features import Voltages
from cardiolens.contracts.signal import SignalRepresentation

R_WINDOW_S = 0.05      # +/- around the R peak for R / S extremes
P_WINDOW_S = (0.20, 0.03)   # P search window before the QRS onset (start, end)


def _extremes(row: np.ndarray, rpeaks: np.ndarray, fs: float, baseline: float) -> tuple[float, float]:
    """(median R above baseline, median |S| below baseline) around the R peaks (µV)."""
    win = int(R_WINDOW_S * fs)
    rs, ss = [], []
    for r in rpeaks:
        seg = row[max(0, r - win): min(len(row), r + win)] - baseline
        if seg.size:
            rs.append(max(0.0, float(seg.max())))
            ss.append(max(0.0, float(-seg.min())))
    return (float(np.median(rs)), float(np.median(ss))) if rs else (0.0, 0.0)


def _baseline(row: np.ndarray, qrs_on: np.ndarray, fs: float) -> float:
    lo, hi = int(0.05 * fs), int(0.01 * fs)
    vals = [np.median(row[max(0, int(a) - lo): max(1, int(a) - hi)]) for a in qrs_on if np.isfinite(a) and a > lo]
    return float(np.median(vals)) if vals else float(np.median(row))


def measure_voltages(sig: SignalRepresentation, rpeaks: np.ndarray, qrs_on: np.ndarray,
                     fs: float, masks: dict[str, np.ndarray] | None = None) -> Voltages:
    v = Voltages()
    if len(rpeaks) == 0:
        return v
    names = set(sig.lead_names)

    def beats(lead: str) -> np.ndarray:
        mk = masks.get(lead) if masks else None
        return rpeaks[mk] if mk is not None and mk.any() else rpeaks

    def rs(lead: str) -> tuple[float, float] | None:
        if lead not in names:
            return None
        row = sig.lead(lead)
        return _extremes(row, beats(lead), fs, _baseline(row, qrs_on, fs))

    for lead, r_attr, s_attr in (("aVL", "r_avl_uv", None), ("V3", None, "s_v3_uv"), ("V5", "r_v5_uv", None),
                                 ("V6", "r_v6_uv", "s_v6_uv"), ("V1", "r_v1_uv", "s_v1_uv"), ("V2", None, "s_v2_uv")):
        e = rs(lead)
        if e is None:
            continue
        if r_attr:
            setattr(v, r_attr, round(e[0], 1))
        if s_attr:
            setattr(v, s_attr, round(e[1], 1))
    limb = [rs(lead) for lead in ("I", "II", "III", "aVL", "aVF") if lead in names]
    if limb:
        v.r_max_limb_uv = round(max(max(e) for e in limb if e is not None), 1)

    # Intrinsicoid deflection: QRS onset -> R peak in V5/V6 (the later of the two).
    idt = []
    for lead in ("V5", "V6"):
        if lead not in names:
            continue
        row = sig.lead(lead)
        win = int(R_WINDOW_S * fs)
        vals = []
        for r, on in zip(rpeaks, qrs_on):
            if not np.isfinite(on):
                continue
            seg = row[max(0, r - win): min(len(row), r + win)]
            if seg.size:
                peak = max(0, r - win) + int(np.argmax(seg))
                vals.append((peak - on) / fs * 1000.0)
        if vals:
            idt.append(float(np.median(vals)))
    if idt:
        v.intrinsicoid_v5_v6_ms = round(max(idt), 1)

    # Terminal P force in V1: negative area (|µV| x ms) of the P wave's terminal part.
    if "V1" in names:
        row = sig.lead("V1")
        a, b = int(P_WINDOW_S[0] * fs), int(P_WINDOW_S[1] * fs)
        forces = []
        for on in qrs_on:
            if not np.isfinite(on) or on < a:
                continue
            seg = row[int(on) - a: int(on) - b]
            base = float(np.median(row[int(on) - a: int(on) - a + int(0.03 * fs)]))
            neg = np.flatnonzero(seg - base < -20.0)
            if neg.size:
                # last negative run before the QRS = terminal force
                end = neg[-1]
                start = end
                while start > 0 and seg[start - 1] - base < -20.0:
                    start -= 1
                depth = float(-(seg[start:end + 1] - base).min())
                forces.append(depth * (end - start + 1) / fs * 1000.0)
            else:
                forces.append(0.0)
        if forces:
            v.p_terminal_v1_uv_ms = round(float(np.median(forces)), 1)
    return v
