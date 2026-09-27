"""Synthetic 12-lead ECG generator (TZ §15 sample; offline determinism).

Produces a deterministic, physiologically-plausible 12-lead ECG so the full
pipeline and tests can run offline (TZ §12 "ядро локально/офлайн"). This is NOT
a clinical model — it is a controllable stimulus for testing detectors. For real
clinical signals use the PTB-XL loader (``ptbxl.py``).

Beats are generated one full RR interval at a time (no overlap), so P/QRS/T are
cleanly delineable and PR/QT track the requested inputs across heart rates.
"""

from __future__ import annotations

import numpy as np

from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Patient,
    PatientContext,
    Provenance,
    SignalRepresentation,
)

# P-wave Gaussian sigma (ms): ~100 ms total duration, as on real ECGs. A narrower P
# has its energy above the atrial band (2-8 Hz) used by the flutter/AV-block
# detector and is not representative.
P_SIGMA_MS = 20.0

# Per-lead QRS scaling (rough, plausible signs/magnitudes).
_LEAD_QRS_SCALE = {
    "I": 0.7, "II": 1.0, "III": 0.5, "aVR": -0.8, "aVL": 0.3, "aVF": 0.8,
    "V1": -0.6, "V2": 0.6, "V3": 1.0, "V4": 1.3, "V5": 1.4, "V6": 1.1,
}


def _beat(
    rr_ms: float, fs: float, qrs_amp_uv: float, t_amp_uv: float,
    qt_ms: float, pr_ms: float, st_shift_uv: float = 0.0,
    p_amp_scale: float = 1.0, qrs_width_scale: float = 1.0,
    terminal_uv: float = 0.0,
) -> np.ndarray:
    """One PQRST complex spanning exactly ``rr_ms`` (µV).

    The P onset is anchored early, the QRS onset sits ``pr_ms`` after it and the
    T offset ``qt_ms`` after the QRS onset, so measured PR/QT track the inputs.
    QT is capped to keep the T wave inside the RR interval. ``qrs_width_scale`` widens
    the QRS (for BBB/WPW), ``terminal_uv`` adds a late terminal deflection (BBB
    morphology, e.g. R' in V1), and ``p_amp_scale`` scales/removes the P wave
    (0 = no P, for atrial fibrillation).
    """
    n = int(round(rr_ms / 1000.0 * fs))
    t = np.linspace(0, rr_ms, n, endpoint=False)

    def gauss(center_ms: float, width_ms: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((t - center_ms) / width_ms) ** 2)

    p_onset = 20.0                     # P wave begins early in the beat
    p_peak = p_onset + 2.0 * P_SIGMA_MS   # onset ~2 sigma before the peak
    w = qrs_width_scale
    qrs_onset = p_onset + pr_ms        # PR = P onset -> QRS onset (physiologic)
    r_pos = qrs_onset + 32.0 * w       # Q wave (centre -14w, sigma 7w) starts here
    max_qt = rr_ms - qrs_onset - 40.0  # keep a short TP gap before the next P
    qt_eff = min(qt_ms, max_qt) if max_qt > 140 else max(140.0, max_qt)
    # Narrower T at short QT (fast rates) so the T does not fuse with the QRS.
    t_sigma = 40.0 * min(1.0, qt_eff / 380.0) ** 0.5
    t_peak = qrs_onset + qt_eff - 2.0 * t_sigma   # T offset (~peak + 2 sigma) = onset + QT

    p = gauss(p_peak, P_SIGMA_MS, 120.0 * p_amp_scale)
    q = gauss(r_pos - 14.0 * w, 7.0 * w, -0.15 * qrs_amp_uv)
    r = gauss(r_pos, 10.0 * w, qrs_amp_uv)
    s = gauss(r_pos + 18.0 * w, 9.0 * w, -0.25 * qrs_amp_uv)
    # Terminal deflection ~40 ms after R (RSR' / broad terminal forces of BBB).
    term = gauss(r_pos + 45.0 * w, 10.0 * w, terminal_uv) if abs(terminal_uv) > 1e-6 else 0.0
    tw = gauss(t_peak, t_sigma, t_amp_uv)
    beat = p + q + r + s + term + tw
    # Optional ST-segment elevation (STEMI stimulus): raise the J-to-T plateau with
    # raised-cosine edges (a rectangular step confuses T-offset delineation).
    if abs(st_shift_uv) > 1e-6:
        j = r_pos + 36.0 * w            # QRS offset
        t_on = t_peak - 1.5 * t_sigma
        ramp = 20.0
        rise = np.clip((t - (j - ramp)) / (2 * ramp), 0.0, 1.0)
        fall = np.clip(((t_on + ramp) - t) / (2 * ramp), 0.0, 1.0)
        window = 0.5 * (1 - np.cos(np.pi * rise)) * 0.5 * (1 - np.cos(np.pi * fall))
        beat = beat + window.astype(np.float32) * st_shift_uv
    return beat.astype(np.float32)


# Relative atrial-wave amplitude per lead (F-waves negative in II/III/aVF, positive
# in V1; P waves upright in II, inverted in aVR).
_FLUTTER_LEAD_UV = {"II": -150.0, "III": -140.0, "aVF": -150.0, "aVR": 120.0, "I": -40.0,
                    "aVL": 30.0, "V1": 100.0, "V2": 60.0, "V3": 40.0, "V4": 30.0,
                    "V5": 30.0, "V6": 30.0}
_P_LEAD_SCALE = {"II": 1.0, "III": 0.6, "aVF": 0.8, "aVR": -0.8, "I": 0.5, "aVL": 0.2,
                 "V1": 0.5, "V2": 0.4, "V3": 0.4, "V4": 0.4, "V5": 0.4, "V6": 0.4}


def _flutter_overlay(n_samples: int, fs: float, atrial_rate_bpm: float):
    """Continuous sawtooth F-waves (no isoelectric baseline) across the record."""
    from scipy.signal import sawtooth

    t = np.arange(n_samples) / fs
    wave = sawtooth(2 * np.pi * atrial_rate_bpm / 60.0 * t, width=0.3)
    k = max(1, int(0.015 * fs))
    wave = np.convolve(wave, np.ones(k) / k, mode="same")  # soften the edges

    def overlay(lead: str) -> np.ndarray:
        return (_FLUTTER_LEAD_UV.get(lead, 30.0) * wave).astype(np.float32)

    return overlay


def _p_train_overlay(n_samples: int, fs: float, atrial_rate_bpm: float):
    """Regular Gaussian P waves at ``atrial_rate_bpm``, independent of the QRS."""
    t_ms = np.arange(n_samples) / fs * 1000.0
    period = 60000.0 / atrial_rate_bpm
    train = np.zeros(n_samples, dtype=np.float64)
    centre = 60.0
    while centre < t_ms[-1] + 50:
        train += 120.0 * np.exp(-0.5 * ((t_ms - centre) / 25.0) ** 2)
        centre += period

    def overlay(lead: str) -> np.ndarray:
        return (_P_LEAD_SCALE.get(lead, 0.4) * train).astype(np.float32)

    return overlay


def generate_synthetic_ecg(
    *,
    heart_rate_bpm: float = 75.0,
    duration_s: float = 10.0,
    sampling_rate_hz: float = 500.0,
    qt_ms: float | None = None,
    pr_ms: float = 160.0,
    t_amp_uv: float = 300.0,
    noise_uv: float = 8.0,
    seed: int = 42,
    age_years: float | None = 40.0,
    sex: str | None = "male",
    potassium: float | None = None,
    st_elevation_uv: float = 0.0,
    st_leads: list[str] | None = None,
    qrs_width_scale: float = 1.0,
    p_amp_scale: float = 1.0,
    rr_jitter_ms: float = 0.0,
    bbb: str | None = None,
    amplitude_scale: float = 1.0,
    flutter_atrial_rate_bpm: float | None = None,
    flutter_conduction: str = "variable",
    dissociated_atrial_rate_bpm: float | None = None,
) -> SignalRepresentation:
    """Generate a deterministic synthetic 12-lead ECG.

    Knobs let tests synthesise a range of patterns: ``qt_ms``/``pr_ms``/
    ``heart_rate_bpm``/``t_amp_uv`` (long-QT, AV block I, brady/tachy, peaked-T);
    ``st_elevation_uv``+``st_leads`` (STEMI territory); ``qrs_width_scale``
    (wide QRS); ``bbb`` in {"rbbb","lbbb"} (bundle branch block morphology);
    ``p_amp_scale=0``+``rr_jitter_ms`` (atrial fibrillation);
    ``flutter_atrial_rate_bpm`` (continuous sawtooth F-waves; ``flutter_conduction``
    "variable" = 2:1/3:1/4:1 mix with AV-delay jitter, "fixed" = one ratio derived
    from ``heart_rate_bpm``); ``dissociated_atrial_rate_bpm`` (an independent P train
    at that rate with the ventricles at ``heart_rate_bpm`` — complete AV block).
    """
    rng = np.random.default_rng(seed)
    fs = float(sampling_rate_hz)
    n_samples = int(round(duration_s * fs))
    rr_ms = 60000.0 / heart_rate_bpm
    if qt_ms is None:
        qt_ms = 400.0 * (rr_ms / 1000.0) ** 0.5

    lead_t_scale = {ln: 1.0 for ln in STANDARD_12_LEADS}
    lead_t_scale["aVR"] = -0.6
    lead_t_scale["V1"] = -0.3

    # BBB: widen QRS and set a terminal deflection whose sign in V1 encodes the
    # block type (RBBB -> terminal positive R' in V1; LBBB -> terminal negative).
    terminal_by_lead = {ln: 0.0 for ln in STANDARD_12_LEADS}
    if bbb in ("rbbb", "lbbb"):
        qrs_width_scale = max(qrs_width_scale, 1.9)
        sign = 1.0 if bbb == "rbbb" else -1.0
        terminal_by_lead["V1"] = sign * 500.0
        terminal_by_lead["V2"] = sign * 400.0
        terminal_by_lead["V6"] = -sign * 350.0
        terminal_by_lead["I"] = -sign * 300.0

    # Shared per-beat RR sequence (consistent across all leads) with jitter.
    beat_rrs: list[float] = []
    total = 0.0
    max_ms = duration_s * 1000.0 + rr_ms
    atrial_overlay = None  # per-lead callable -> µV overlay across the whole record
    if flutter_atrial_rate_bpm:
        p_amp_scale = 0.0
        f_period = 60000.0 / flutter_atrial_rate_bpm
        fixed_k = max(2, int(round(flutter_atrial_rate_bpm / heart_rate_bpm)))
        while total < max_ms:
            k = fixed_k if flutter_conduction == "fixed" else int(rng.choice([2, 3, 3, 4]))
            # Variable conduction: the AV delay wanders (Wenckebach-like), so the
            # F-wave phase at each QRS is not fixed.
            jit = 0.0 if flutter_conduction == "fixed" else rng.uniform(-0.35, 0.35) * f_period
            this_rr = max(300.0, k * f_period + jit)
            beat_rrs.append(this_rr)
            total += this_rr
        atrial_overlay = _flutter_overlay(n_samples, fs, flutter_atrial_rate_bpm)
    else:
        while total < max_ms:
            jit = rng.uniform(-rr_jitter_ms, rr_jitter_ms) if rr_jitter_ms > 0 else 0.0
            this_rr = max(300.0, rr_ms + jit)
            beat_rrs.append(this_rr)
            total += this_rr
    if dissociated_atrial_rate_bpm:
        p_amp_scale = 0.0
        atrial_overlay = _p_train_overlay(n_samples, fs, dissociated_atrial_rate_bpm)

    signal = np.zeros((12, n_samples), dtype=np.float32)
    base_qrs_uv = 900.0 * amplitude_scale
    t_amp_uv = t_amp_uv * amplitude_scale
    st_target = set(st_leads or [])

    for li, lead in enumerate(STANDARD_12_LEADS):
        row = np.zeros(n_samples, dtype=np.float32)
        start = 0
        for this_rr in beat_rrs:
            beat = _beat(
                this_rr, fs,
                qrs_amp_uv=base_qrs_uv * _LEAD_QRS_SCALE[lead],
                t_amp_uv=t_amp_uv * lead_t_scale[lead],
                qt_ms=qt_ms, pr_ms=pr_ms,
                st_shift_uv=st_elevation_uv if lead in st_target else 0.0,
                p_amp_scale=p_amp_scale, qrs_width_scale=qrs_width_scale,
                terminal_uv=terminal_by_lead[lead],
            )
            end = min(start + len(beat), n_samples)
            if start >= n_samples:
                break
            row[start:end] = beat[: end - start]
            start += len(beat)
        t_axis = np.arange(n_samples) / fs
        wander = 20.0 * np.sin(2 * np.pi * 0.15 * t_axis)
        row = row + wander + rng.normal(0.0, noise_uv, n_samples)
        if atrial_overlay is not None:
            row = row + atrial_overlay(lead) * amplitude_scale
        signal[li] = row.astype(np.float32)

    return SignalRepresentation(
        signal=signal,
        sampling_rate_hz=fs,
        lead_names=list(STANDARD_12_LEADS),
        duration_s=float(duration_s),
        source="synthetic",
        patient=Patient(age_years=age_years, sex=sex),
        context=PatientContext(potassium=potassium),
        provenance=Provenance(digitized=False, confidence=1.0),
    )
