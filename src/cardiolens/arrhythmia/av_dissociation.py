"""Atrial activity not linked to the QRS — AV dissociation / complete AV block (D117).

In complete AV block the P waves march through the cardiac cycle at their own rate, so
they average out of the median QRS-T. Subtracting the median QRS-T from every beat
leaves them in the residual as a regular series of peaks faster than the ventricles.
A P wave locked to the QRS (sinus rhythm, first-degree or 2:1 block) is part of the
median and leaves no residual.

Developed on ZZU-pECG dev children (D117): the residual peak size separates complete AV
block (median 212 µV) from other ECGs (57 µV); premature atrial beats also leave
unlocked P waves, but make the ventricular rhythm irregular, so a regular ventricular
rhythm is required.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.signal import find_peaks

_PRE_S = 0.25          # beat window starts this long before R
_QRS_MASK_S = (0.06, 0.10)   # subtraction residue around R that is not atrial
_MIN_CORR = 0.8        # beats unlike the median (ectopy, artefact) are left out
_MIN_PEAK_UV = 25.0
_MIN_PEAK_NOISE = 4.0
_MIN_PP_S = 0.2


def atrial_residual(x: np.ndarray, rpeaks, fs: float) -> dict | None:
    """Peaks of the dominant polarity left after subtracting the median QRS-T from each
    beat of one lead. Returns the atrial rate (per minute), the median peak prominence
    (µV), the residual noise (µV) and the robust spread of P-P intervals (MAD / median),
    or None when the strip is too short."""
    rp = np.asarray(rpeaks, dtype=int)
    rr = np.diff(rp)
    pre = int(_PRE_S * fs)
    length = int(rr.min()) if rr.size else 0
    if rr.size < 3 or length < int(0.3 * fs):
        return None
    idx = [r for r in rp[:-1] if r - pre >= 0 and r - pre + length <= x.size]
    if len(idx) < 3:
        return None
    segs = np.stack([x[r - pre:r - pre + length] for r in idx]).astype(float)
    tpl = np.median(segs, axis=0)
    q0, q1 = pre - int(_QRS_MASK_S[0] * fs), pre + int(_QRS_MASK_S[1] * fs)
    res = []
    for s in segs:
        if np.corrcoef(s, tpl)[0, 1] < _MIN_CORR:
            continue
        d = s - tpl
        d[q0:q1] = 0.0
        res.append(d)
    if len(res) < 3:
        return None
    resid = np.concatenate(res)
    noise = 1.4826 * float(np.median(np.abs(resid - np.median(resid)))) + 1e-9
    best = None
    for sign in (1.0, -1.0):
        pk, pr = find_peaks(sign * resid, prominence=max(_MIN_PEAK_NOISE * noise, _MIN_PEAK_UV),
                            distance=int(_MIN_PP_S * fs))
        if best is None or len(pk) > len(best[0]):
            best = (pk, pr)
    pk, pr = best
    # P-P intervals only inside one beat window: windows are cut at the shortest RR.
    pp = np.asarray([b - a for a, b in pairwise(pk) if a // length == b // length]) / fs
    pp_spread = (float(1.4826 * np.median(np.abs(pp - np.median(pp))) / np.median(pp))
                 if pp.size >= 3 else None)
    return {
        "atrial_rate": len(pk) / (len(res) * length / fs) * 60.0,
        "peak_uv": float(np.median(pr["prominences"])) if len(pk) else 0.0,
        "noise_uv": noise,
        "pp_spread": pp_spread,
    }


def av_dissociation(sig, rpeaks, heart_rate_bpm: float | None, rr_mad_cv: float | None,
                    cfg) -> tuple[str, dict] | None:
    """(lead, features) of the first lead (II, then V1) showing unlinked regular atrial
    activity on a regular ventricular rhythm, else None."""
    if heart_rate_bpm is None or rr_mad_cv is None or rr_mad_cv > cfg.rr_mad_cv_max:
        return None
    for lead in ("II", "V1"):
        if lead not in sig.lead_names:
            continue
        f = atrial_residual(sig.lead(lead).astype(float), rpeaks, sig.sampling_rate_hz)
        if f is None or f["pp_spread"] is None:
            continue
        if (f["atrial_rate"] >= cfg.atrial_ventricular_ratio * heart_rate_bpm
                and f["pp_spread"] <= cfg.pp_spread_max
                and f["peak_uv"] >= cfg.peak_uv_min
                and f["peak_uv"] >= cfg.peak_noise_min * f["noise_uv"]):
            return lead, f
    return None
