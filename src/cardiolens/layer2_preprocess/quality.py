"""Layer 2 — signal quality and electrode checks (TZ §5).

Per-lead SQI, lead-off detection, and LA/RA electrode-reversal detection. The
reversal detector emits a *hypothesis* (finding code ``lead_reversal_la_ra``)
that Layer 5 treats as an artifact explanation (TZ §5, §8.3) rather than
fabricating pathology.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import Quality, SignalRepresentation


def per_lead_sqi(sig: SignalRepresentation) -> list[float]:
    """Heuristic per-lead SQI in [0, 1] (TZ §5).

    Combines: flatline penalty (lead-off), and a crude SNR proxy from the ratio
    of in-band structured energy to high-frequency jitter.
    """
    scores: list[float] = []
    for i in range(sig.n_leads):
        row = sig.signal[i].astype(np.float64)
        ptp = float(np.ptp(row))
        if ptp < 10.0:  # <10 µV span -> essentially flat (lead-off)
            scores.append(0.0)
            continue
        # High-frequency noise proxy: std of sample-to-sample difference.
        hf = float(np.std(np.diff(row)))
        sig_std = float(np.std(row)) + 1e-9
        snr = sig_std / (hf + 1e-9)
        score = float(np.clip(snr / (snr + 1.0), 0.0, 1.0))
        scores.append(round(score, 3))
    return scores


def detect_lead_off(sqi: list[float], threshold: float = 0.05) -> list[str]:
    """Flag near-flat leads as lead-off (TZ §5)."""
    return [f"lead_off:{i}" for i, s in enumerate(sqi) if s <= threshold]


def detect_la_ra_reversal(sig: SignalRepresentation) -> bool:
    """Detect LA/RA electrode reversal (TZ §5).

    Classic signature: lead I becomes globally inverted and aVR looks like a
    normal lead. We test for a predominantly negative P/QRS in lead I combined
    with an upright aVR — a predictable, non-physiological pattern (TZ §8.3).
    Requires the standard limb leads to be present.
    """
    names = sig.lead_names
    if not {"I", "aVR"}.issubset(set(names)):
        return False
    lead_i = sig.lead("I").astype(np.float64)
    avr = sig.lead("aVR").astype(np.float64)
    # Dominant (largest-magnitude) deflection sign, robust to baseline.
    i_dominant = lead_i[np.argmax(np.abs(lead_i - np.median(lead_i)))] - np.median(lead_i)
    avr_dominant = avr[np.argmax(np.abs(avr - np.median(avr)))] - np.median(avr)
    # Normal: lead I dominant deflection positive, aVR negative.
    # Reversal: lead I dominant negative AND aVR dominant positive.
    return bool(i_dominant < 0 and avr_dominant > 0)


def assess_quality(
    sig: SignalRepresentation, *, sqi_accept_threshold: float
) -> tuple[Quality, bool]:
    """Populate ``sig.quality`` and return (quality, la_ra_reversed)."""
    sqi = per_lead_sqi(sig)
    flags: list[str] = []
    flags.extend(detect_lead_off(sqi))
    low = [i for i, s in enumerate(sqi) if s < sqi_accept_threshold]
    if low:
        flags.append(f"low_sqi_leads:{','.join(map(str, low))}")

    reversed_la_ra = detect_la_ra_reversal(sig)
    if reversed_la_ra:
        flags.append("suspected_la_ra_reversal")

    quality = Quality(per_lead_sqi=sqi, flags=flags)
    sig.quality = quality
    return quality, reversed_la_ra
