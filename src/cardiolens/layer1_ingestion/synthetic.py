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

    P peak is anchored early; R sits ``pr_ms`` after it (so measured PR tracks
    the input); the T wave is placed so measured QT tracks ``qt_ms``. QT is
    capped to keep the T wave inside the RR interval. ``qrs_width_scale`` widens
    the QRS (for BBB/WPW), ``terminal_uv`` adds a late terminal deflection (BBB
    morphology, e.g. R' in V1), and ``p_amp_scale`` scales/removes the P wave
    (0 = no P, for atrial fibrillation).
    """
    n = int(round(rr_ms / 1000.0 * fs))
    t = np.linspace(0, rr_ms, n, endpoint=False)

    def gauss(center_ms: float, width_ms: float, amp: float) -> np.ndarray:
        return amp * np.exp(-0.5 * ((t - center_ms) / width_ms) ** 2)

    p_onset = 20.0                     # P wave begins early in the beat
    p_peak = p_onset + 30.0
    r_pos = p_onset + pr_ms            # QRS onset ≈ pr_ms after P onset
    max_qt = rr_ms - r_pos - 60.0
    qt_eff = min(qt_ms, max_qt) if max_qt > 140 else max(140.0, max_qt)
    t_peak = r_pos + qt_eff - 100.0    # so T offset (~t_peak+80) ≈ onset+qt
    w = qrs_width_scale

    p = gauss(p_peak, 12.0, 120.0 * p_amp_scale)
    q = gauss(r_pos - 14.0 * w, 7.0 * w, -0.15 * qrs_amp_uv)
    r = gauss(r_pos, 10.0 * w, qrs_amp_uv)
    s = gauss(r_pos + 18.0 * w, 9.0 * w, -0.25 * qrs_amp_uv)
    # Terminal deflection ~40 ms after R (RSR' / broad terminal forces of BBB).
    term = gauss(r_pos + 45.0 * w, 10.0 * w, terminal_uv) if abs(terminal_uv) > 1e-6 else 0.0
    tw = gauss(t_peak, 40.0, t_amp_uv)
    beat = p + q + r + s + term + tw
    # Optional ST-segment elevation (STEMI stimulus): raise the J-to-T plateau.
    if abs(st_shift_uv) > 1e-6:
        j = r_pos + 30.0            # just after S
        t_on = t_peak - 60.0
        mask = (t >= j) & (t <= t_on)
        beat = beat + mask.astype(np.float32) * st_shift_uv
    return beat.astype(np.float32)


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
) -> SignalRepresentation:
    """Generate a deterministic synthetic 12-lead ECG.

    Knobs let tests synthesise a range of patterns: ``qt_ms``/``pr_ms``/
    ``heart_rate_bpm``/``t_amp_uv`` (long-QT, AV block I, brady/tachy, peaked-T);
    ``st_elevation_uv``+``st_leads`` (STEMI territory); ``qrs_width_scale``
    (wide QRS); ``bbb`` in {"rbbb","lbbb"} (bundle branch block morphology);
    ``p_amp_scale=0``+``rr_jitter_ms`` (atrial fibrillation).
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
    while total < max_ms:
        jit = rng.uniform(-rr_jitter_ms, rr_jitter_ms) if rr_jitter_ms > 0 else 0.0
        this_rr = max(300.0, rr_ms + jit)
        beat_rrs.append(this_rr)
        total += this_rr

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
