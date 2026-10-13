"""Pacemaker spikes on the RAW signal (TZ §7.1).

Ported from claude/vibrant-curie-ro1zsz (D24 spike-QRS coupling, D74-D75 multi-lead
spike by second difference). Runs before the Layer-2 band-pass, which erases spikes.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import SignalRepresentation


def pacing_spike_ratio(sig: SignalRepresentation) -> float:
    """Pacing-spike prominence on the RAW signal (§7.1).

    A pacemaker spike is a very narrow, high-slew deflection. Bandpass filtering
    (Layer 2) removes it, so this runs on the raw signal. Returns the max across
    leads of (largest single-sample |diff|) / (99th-percentile |diff|): a pacer
    spike is an extreme outlier in the slope distribution (~9 for paced records
    vs ~2.6 for normal on PTB-XL). 0 when not computable.
    """
    best = 0.0
    for name in sig.lead_names:
        x = sig.lead(name).astype(np.float64)
        d = np.abs(np.diff(x))
        if d.size < 10:
            continue
        p99 = float(np.percentile(d, 99))
        if p99 > 0:
            best = max(best, float(d.max()) / p99)
    return round(best, 2)


def pacing_spike_times(sig: SignalRepresentation) -> list[float]:
    """Times (s) of pacing-spike candidates on the RAW signal (§7.1).

    Uses the lead with the most extreme slope outlier; a candidate is a sample
    step > 4× the lead's 99th-percentile |diff| and ≥ 0.3× its largest step
    (smaller spikes of the same pacer count). Candidates < 10 ms apart merge.
    """
    best, x_best = 0.0, None
    for name in sig.lead_names:
        d = np.abs(np.diff(sig.lead(name).astype(np.float64)))
        if d.size < 10:
            continue
        p99 = float(np.percentile(d, 99))
        if p99 > 0 and float(d.max()) / p99 > best:
            best, x_best = float(d.max()) / p99, d
    if x_best is None:
        return []
    d = x_best
    p99 = float(np.percentile(d, 99))
    idx = np.flatnonzero((d > 4 * p99) & (d >= 0.3 * d.max()))
    fs = sig.sampling_rate_hz
    merged = [int(i) for k, i in enumerate(idx) if k == 0 or i - idx[k - 1] > 0.01 * fs]
    return [i / fs for i in merged]


def pacing_multilead_spike_times(sig: SignalRepresentation, k: float = 3.0,
                                 min_leads: int = 6) -> list[float]:
    """Times (s) of a spike-shaped event seen by >= ``min_leads`` leads at once, RAW signal (D75).

    A pacer spike is one narrow up-and-back event seen by every lead. Per lead, the
    absolute second difference is scaled by its 99th percentile; a candidate is a
    sample where >= ``min_leads`` leads exceed ``k`` (±1 sample of inter-lead jitter).
    The second difference (not the slope) keeps out the steep but smooth upstroke of a
    narrow QRS, which the slope version (D74) mistook for a spike. Candidates < 10 ms
    apart merge. Development: PTB-XL paced 180 -> 209 of 266 found, no new false calls
    on PTB-XL, Georgia (3 samples) or CPSC non-paced records.
    """
    if len(sig.lead_names) < min_leads:
        return []
    x = np.asarray(sig.signal, dtype=np.float64)
    if x.shape[1] < 12:
        return []
    d2 = np.abs(x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2])
    d2 = np.pad(d2, ((0, 0), (1, 0)))
    p99 = np.percentile(d2, 99, axis=1, keepdims=True)
    p99[p99 <= 0] = np.inf
    z = d2 / p99
    z = np.maximum(z, np.maximum(np.roll(z, 1, axis=1), np.roll(z, -1, axis=1)))
    idx = np.flatnonzero((z > k).sum(axis=0) >= min_leads)
    fs = sig.sampling_rate_hz
    merged = [int(i) for j, i in enumerate(idx) if j == 0 or i - idx[j - 1] > 0.01 * fs]
    return [i / fs for i in merged]


def pacing_spike_coupling(spike_times_s: list[float], r_peak_times_s: list[float],
                          window_s: float = 0.2) -> float:
    """Fraction of beats with a spike candidate 5 ms – ``window_s`` before the R peak.

    A pacemaker spike is time-locked to the paced QRS; an artifact spike lands
    at random. On development data (Chapman/Ningbo + PTB-XL outside fold 10) no
    non-paced record with an extreme slope outlier exceeded 0.10 (DECISIONS D24).
    """
    if not r_peak_times_s:
        return 0.0
    sp = np.asarray(spike_times_s, dtype=float)
    hits = [bool(np.any((sp >= r - window_s) & (sp <= r - 0.005))) for r in r_peak_times_s]
    return round(float(np.mean(hits)), 3)
