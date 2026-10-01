"""Atrial activity before the QRS (D42-D45).

In sinus rhythm every conducted beat is preceded by the same P wave, so the stretch
of signal before each non-premature QRS repeats from beat to beat and carries a
visible deflection. Flutter, junctional and paced rhythms (and AF) do not show that
organised P wave. The automated P-wave delineator cannot tell these apart — it
labels a "P" before almost every beat (D15a) — but the beat-to-beat repeatability
and the size of the pre-QRS window can.
"""

from __future__ import annotations

import numpy as np

_LEADS = ("II", "V1")
_WIN_START_S = 0.30          # window [R - 300 ms, R - 60 ms]
_WIN_END_S = 0.06
_DEFAULT_QT_S = 0.40         # the window starts after the previous beat's QT
_NORMAL_PRE_MIN = 0.90       # non-premature beat: pre-RR >= 0.90 x median RR
_MIN_BEATS = 4

# A rate-based "sinus" label is replaced when lead II shows no organised P wave:
# consistency at most P_CONSISTENCY_MAX or P-window amplitude at most
# P_AMPLITUDE_MAX x QRS (D45; development data: Chapman, PTB-XL — 33% of non-sinus
# rhythms called sinus, 0.9% of true sinus).
P_CONSISTENCY_MAX = 0.30
P_AMPLITUDE_MAX = 0.03
_P_WIN = (0.25, 0.06)          # amplitude window [R - 250 ms, R - 60 ms]

# An AF call keeps its label but is hedged when both II and V1 repeat the pre-QRS
# window at least this well (D51): the irregularity may be sinus rhythm with
# ectopy. Development data (D42/D43 sets): 95% of hedged AF calls had no AF label,
# 2.1% of true AF calls were hedged; held-out Georgia AF: 1.7%. Withholding AF instead cost 3-5% AF
# sensitivity on held-out data (D42), so the call itself is left unchanged.
AF_HEDGE_MIN = 0.75


def pre_qrs_consistency(sig, rpeaks, qt_ms: float | None = None) -> tuple[float | None, dict[str, float]]:
    """Lowest over II/V1 of the median leave-one-out correlation of pre-QRS windows.

    Only non-premature beats whose window starts after the previous beat's QT
    count: at fast rates the previous T wave falls in the window and repeats
    like a P wave would. Returns ``(value, per_lead)``; ``value`` is None when
    too few beats qualify.
    """
    fs = sig.sampling_rate_hz
    rp = np.asarray(rpeaks, dtype=int)
    per_lead: dict[str, float] = {}
    if rp.size < 5:
        return None, per_lead
    rr = np.diff(rp)
    med = float(np.median(rr))
    qt_s = qt_ms / 1000.0 if qt_ms else _DEFAULT_QT_S
    a, b, gap = int(_WIN_START_S * fs), int(_WIN_END_S * fs), int(qt_s * fs)
    idx = [i for i in range(1, len(rp))
           if rp[i] - a >= 0 and rr[i - 1] >= _NORMAL_PRE_MIN * med and rp[i] - a > rp[i - 1] + gap]
    if len(idx) < _MIN_BEATS:
        return None, per_lead
    for lead in _LEADS:
        if lead not in sig.lead_names:
            continue
        x = sig.lead(lead).astype(float)
        windows = np.stack([x[rp[i] - a:rp[i] - b] for i in idx])
        windows = windows - windows.mean(axis=1, keepdims=True)
        total = windows.sum(axis=0)
        corrs = []
        for w in windows:
            rest = (total - w) / (len(windows) - 1)
            denom = float(np.std(w) * np.std(rest))
            corrs.append(float(np.mean((w - w.mean()) * (rest - rest.mean()))) / denom if denom > 0 else 0.0)
        per_lead[lead] = round(float(np.median(corrs)), 3)
    if not per_lead:
        return None, per_lead
    return round(float(min(per_lead.values())), 3), per_lead


def p_window_amplitude(sig, rpeaks, lead: str = "II") -> float | None:
    """Peak-to-peak of the beat-averaged pre-QRS window relative to the QRS."""
    fs = sig.sampling_rate_hz
    if lead not in sig.lead_names:
        return None
    rp = np.asarray(rpeaks, dtype=int)
    a, b, q = int(_P_WIN[0] * fs), int(_P_WIN[1] * fs), int(0.05 * fs)
    ok = [r for r in rp[1:] if r - a >= 0 and r + q < sig.signal.shape[1]]
    if len(ok) < 3:
        return None
    x = sig.lead(lead).astype(float)
    avg = np.mean([x[r - a:r - b] - x[r - a:r - b].mean() for r in ok], axis=0)
    qrs = float(np.median([np.ptp(x[r - q:r + q]) for r in ok]))
    return round(float(np.ptp(avg) / max(qrs, 1.0)), 3)


# Atrial flutter among AF calls (D46, D53): after cancelling the QRST, flutter leaves a
# regular F-wave train (a sharp spectral peak), AF a broad f-wave spectrum. Renaming AF
# to flutter on this index failed held-out checks (D46: 7.5% of true AF renamed), so an
# AF call is only hedged. Development data (D53, 2 140 AF calls; Ningbo flutter labels
# excluded, D52): >= 0.65 hedges 27% of flutter and 1.4% of true AF (2.6% at most per
# source); held-out: 22% of flutter, 1.3-2.1% of true AF.
_ORG_LEADS = ("II", "III", "aVF", "V1")
FLUTTER_HEDGE_MIN = 0.65


def _qrst_cancelled(x: np.ndarray, rpeaks: np.ndarray, fs: float) -> np.ndarray | None:
    """Median-beat QRST cancellation with per-beat least-squares scaling; QRS blanked."""
    a, b = int(0.25 * fs), int(0.45 * fs)
    ok = [r for r in rpeaks if r - a >= 0 and r + b < len(x)]
    if len(ok) < 4:
        return None
    beats = np.stack([x[r - a:r + b] for r in ok])
    tmpl = np.median(beats, axis=0)
    energy = float(tmpl @ tmpl)
    res = x.astype(float).copy()
    for r, beat in zip(ok, beats, strict=True):
        k = float(beat @ tmpl) / energy if energy > 0 else 1.0
        res[r - a:r + b] = beat - k * tmpl
    h = int(0.06 * fs)
    for r in rpeaks:
        res[max(0, r - h):min(len(res), r + h)] = 0.0
    return res


def atrial_organisation(sig, rpeaks) -> tuple[float | None, float | None]:
    """(organisation index, dominant atrial frequency Hz) of the QRST-cancelled residual.

    Organisation = power within 0.5 Hz of the dominant 3-12 Hz peak and its second
    harmonic over all 3-12 Hz power, averaged over II, III, aVF and V1.
    """
    from scipy.signal import butter, filtfilt, welch

    fs = sig.sampling_rate_hz
    rp = np.asarray(rpeaks, dtype=int)
    if rp.size < 5:
        return None, None
    bb, aa = butter(3, [3 / (fs / 2), 12 / (fs / 2)], btype="band")
    ois, dfs = [], []
    for lead in _ORG_LEADS:
        if lead not in sig.lead_names:
            continue
        res = _qrst_cancelled(sig.lead(lead).astype(float), rp, fs)
        if res is None:
            continue
        y = filtfilt(bb, aa, res)
        f, p = welch(y, fs, nperseg=min(len(y), int(4 * fs)), noverlap=int(2 * fs))
        band = (f >= 3) & (f <= 12)
        if p[band].sum() <= 0:
            continue
        fb, pb = f[band], p[band]
        df = float(fb[np.argmax(pb)])
        near = (np.abs(fb - df) <= 0.5) | (np.abs(fb - 2 * df) <= 0.5)
        ois.append(float(pb[near].sum() / pb.sum()))
        dfs.append(df)
    if not ois:
        return None, None
    return round(float(np.mean(ois)), 3), round(float(np.median(dfs)), 2)


# Atrial flutter with 2:1 conduction read as sinus (D54). The F waves are locked to the
# QRS, so the pre-QRS window repeats like a sinus P (D45 does not catch it). A logistic
# model on lead II separates it from true sinus rhythm/tachycardia (development data:
# original Chapman, 82 flutter called sinus vs 580 sinus; out-of-fold AUC 0.89; at the
# threshold 2% of sinus and ~50% of such flutter are flagged). Feature: (impute median,
# mean, std, coefficient) on the standardised value.
_FLUTTER_2TO1_MODEL = {
    "pc_II": (0.956, 0.93453, 0.10093, -0.4541),
    "rr_cv": (0.017, 0.03463, 0.05893, 0.9817),
    "flat_II": (0.0058, 0.00731, 0.00612, 0.2374),
    "psign_II": (0.11785, 0.10838, 0.17194, -0.9044),
    "h2_II": (0.05625, 0.11334, 0.15264, 0.8634),
    "hr": (100.65, 94.62492, 23.03604, 0.1289),
}
_FLUTTER_2TO1_INTERCEPT = -0.7786
FLUTTER_2TO1_THRESHOLD = 1.5475


def two_to_one_features(sig, rpeaks, lead: str = "II") -> dict[str, float | None]:
    """Lead-II morphology of the cycle on the median beat and the raw-lead spectrum.

    ``flat``: flattest 30 ms of the cycle outside QRS +-60 ms over the QRS size (flutter
    has no isoelectric baseline); ``psign``: signed extreme of [R-250, R-60 ms] against
    the PR baseline over the QRS size (typical F waves are negative in II); ``h2``:
    spectral share at twice the ventricular rate within 3-12 Hz.
    """
    from cardiolens.layer3_measure.global_beat import _pr_baseline, median_beats

    out: dict[str, float | None] = {"flat": None, "psign": None, "h2": None}
    fs = sig.sampling_rate_hz
    rp = np.asarray(rpeaks, dtype=int)
    if rp.size < 5 or lead not in sig.lead_names:
        return out
    rr = float(np.median(np.diff(rp)))
    X, a = median_beats(sig.signal, rp, fs)
    if X is None:
        return out
    x = X[list(sig.lead_names).index(lead)].astype(float)
    w, q = max(2, int(0.03 * fs)), int(0.06 * fs)
    qrs = float(np.ptp(x[a - q:a + q])) or 1.0
    seg = x[a - int(0.25 * fs):a - int(0.06 * fs)] - _pr_baseline(x, a, fs)
    out["psign"] = round(float(seg[np.argmax(np.abs(seg))]) / qrs, 4)
    lo, hi = max(0, a - int(rr) // 2), min(len(x), a + int(rr) // 2)
    cyc = [s for s in range(lo, hi - w) if not (a - q <= s <= a + q or a - q <= s + w <= a + q)]
    if cyc:
        out["flat"] = round(min(float(np.ptp(x[s:s + w])) for s in cyc) / qrs, 4)
    y = sig.lead(lead).astype(float)
    y = y - y.mean()
    power = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / fs)
    band = (f >= 3) & (f <= 12)
    if band.any():
        h2 = band & (np.abs(f - 2 * fs / rr) <= 0.25)
        out["h2"] = round(float(power[h2].sum() / max(power[band].sum(), 1e-12)), 4)
    return out


def flutter_2to1_score(values: dict[str, float | None]) -> float:
    """Linear score of the D54 model; above FLUTTER_2TO1_THRESHOLD -> hedge."""
    z = _FLUTTER_2TO1_INTERCEPT
    for name, (med, mean, std, coef) in _FLUTTER_2TO1_MODEL.items():
        v = values.get(name)
        v = med if v is None else float(v)
        z += coef * (v - mean) / std
    return round(z, 4)
