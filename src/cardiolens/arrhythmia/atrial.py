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
# 2.1% of true AF calls were hedged. Withholding AF instead cost 3-5% AF
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
