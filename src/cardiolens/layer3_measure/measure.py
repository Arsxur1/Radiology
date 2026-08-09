"""Layer 3 — detection and measurements (TZ §6).

QRS detection and P/QRS/T delineation via neurokit2, interval measurement
(PR/QRS/QT + QTc Bazett/Fridericia/Framingham), heart rate / regularity / HRV,
frontal QRS axis, ST measurement at J+60/80 ms, Sokolow-Lyon voltage, and simple
morphology descriptors (peaked/flat/inverted T, U waves).

Where delineation is unreliable we record a note and leave the field ``None``
rather than fabricating a value (TZ §0).
"""

from __future__ import annotations

import warnings

import numpy as np

from cardiolens.contracts.features import (
    Axes,
    Intervals,
    Measurements,
    Morphology,
    STMeasurement,
)
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer3_measure.intervals import (
    qtc_bazett,
    qtc_framingham,
    qtc_fridericia,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Lead preferred for rhythm/delineation.
_RHYTHM_LEAD_PREFERENCE = ["II", "V5", "V2", "I"]


def _pick_rhythm_lead(sig: SignalRepresentation) -> int:
    for name in _RHYTHM_LEAD_PREFERENCE:
        if name in sig.lead_names:
            return sig.lead_names.index(name)
    return 0


def _median_ms(values: list[float], fs: float) -> float | None:
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.median(vals) / fs * 1000.0)


def _delineate(cleaned: np.ndarray, rpeaks: np.ndarray, fs: float):
    """Return per-beat onset/offset arrays via neurokit2 (best-effort)."""
    import neurokit2 as nk

    try:
        _, waves = nk.ecg_delineate(
            cleaned, {"ECG_R_Peaks": rpeaks}, sampling_rate=fs, method="dwt"
        )
        return waves
    except Exception:
        return None


def measure(sig: SignalRepresentation) -> Measurements:
    """Full Layer-3 measurement pass (TZ §6)."""
    import neurokit2 as nk

    fs = sig.sampling_rate_hz
    m = Measurements()
    lead_idx = _pick_rhythm_lead(sig)
    cleaned = sig.signal[lead_idx].astype(np.float64)

    # --- R-peaks, HR, regularity, HRV ---
    try:
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        rpeaks = np.array([], dtype=int)

    m.r_peaks_samples = rpeaks.tolist()
    m.n_beats = int(len(rpeaks))

    rr_s_mean = None
    if len(rpeaks) >= 2:
        rr = np.diff(rpeaks) / fs  # seconds
        rr_s_mean = float(np.mean(rr))
        m.heart_rate_bpm = round(60.0 / rr_s_mean, 1)
        m.rr_std_ms = round(float(np.std(rr) * 1000.0), 1)
        m.hrv_sdnn_ms = round(float(np.std(rr) * 1000.0), 1)
        # Regularity: coefficient of variation of RR < 10% -> regular.
        cv = float(np.std(rr) / (np.mean(rr) + 1e-9))
        m.rhythm_regular = bool(cv < 0.10)
    else:
        m.notes.append("insufficient R-peaks for rhythm analysis (low confidence)")

    # --- Delineation -> intervals ---
    if len(rpeaks) >= 3:
        waves = _delineate(cleaned, rpeaks, fs)
        if waves is not None:
            p_on = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
            qrs_on = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
            qrs_off = np.asarray(waves.get("ECG_R_Offsets", []), dtype=float)
            t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)

            n = min(len(qrs_on), len(qrs_off), len(t_off))
            pr_list, qrs_list, qt_list = [], [], []
            for i in range(n):
                if i < len(p_on) and np.isfinite(p_on[i]) and np.isfinite(qrs_on[i]):
                    pr_list.append(qrs_on[i] - p_on[i])
                if np.isfinite(qrs_on[i]) and np.isfinite(qrs_off[i]):
                    qrs_list.append(qrs_off[i] - qrs_on[i])
                if np.isfinite(qrs_on[i]) and np.isfinite(t_off[i]):
                    qt_list.append(t_off[i] - qrs_on[i])

            m.intervals = Intervals(
                pr_ms=_median_ms(pr_list, fs),
                qrs_ms=_median_ms(qrs_list, fs),
                qt_ms=_median_ms(qt_list, fs),
            )
            # ST measured off QRS offsets (J points), baseline from PR segment.
            _measure_st(sig, qrs_on, qrs_off, m)
        else:
            m.notes.append("delineation failed (low confidence intervals)")

    # --- QTc ---
    if m.intervals.qt_ms is not None and rr_s_mean:
        m.intervals.qtc_bazett_ms = round(qtc_bazett(m.intervals.qt_ms, rr_s_mean), 1)
        m.intervals.qtc_fridericia_ms = round(
            qtc_fridericia(m.intervals.qt_ms, rr_s_mean), 1
        )
        m.intervals.qtc_framingham_ms = round(
            qtc_framingham(m.intervals.qt_ms, rr_s_mean), 1
        )

    # --- Frontal QRS axis ---
    m.axes = Axes(qrs_deg=_qrs_axis(sig, rpeaks, fs))

    # --- Sokolow-Lyon voltage (LVH) ---
    m.sokolow_lyon_uv = _sokolow_lyon(sig, rpeaks, fs)

    # --- Morphology (peaked/flat/inverted T) ---
    m.morphology.st = m.morphology.st  # populated by _measure_st
    _measure_t_morphology(sig, rpeaks, fs, m)

    return m


def _qrs_amplitude(row: np.ndarray, rpeaks: np.ndarray, fs: float) -> float:
    """Net QRS deflection = median signed peak amplitude at R locations (µV)."""
    if len(rpeaks) == 0:
        return 0.0
    win = int(0.05 * fs)
    vals = []
    for r in rpeaks:
        a, b = max(0, r - win), min(len(row), r + win)
        seg = row[a:b]
        # Signed extreme (largest absolute deflection keeps its sign).
        vals.append(seg[np.argmax(np.abs(seg))])
    return float(np.median(vals))


def _qrs_axis(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    if not {"I", "aVF"}.issubset(set(sig.lead_names)) or len(rpeaks) == 0:
        return None
    i_amp = _qrs_amplitude(sig.lead("I"), rpeaks, fs)
    f_amp = _qrs_amplitude(sig.lead("aVF"), rpeaks, fs)
    return round(float(np.degrees(np.arctan2(f_amp, i_amp))), 1)


def _sokolow_lyon(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    names = set(sig.lead_names)
    if not {"V1", "V5", "V6"}.issubset(names) or len(rpeaks) == 0:
        return None
    win = int(0.05 * fs)

    def deep_s(lead: str) -> float:
        row = sig.lead(lead)
        vals = []
        for r in rpeaks:
            a, b = max(0, r - win), min(len(row), r + win)
            vals.append(abs(min(row[a:b])))
        return float(np.median(vals))

    def tall_r(lead: str) -> float:
        row = sig.lead(lead)
        vals = []
        for r in rpeaks:
            a, b = max(0, r - win), min(len(row), r + win)
            vals.append(max(row[a:b]))
        return float(np.median(vals))

    return round(deep_s("V1") + max(tall_r("V5"), tall_r("V6")), 1)


def _measure_st(
    sig: SignalRepresentation,
    qrs_on: np.ndarray,
    qrs_off: np.ndarray,
    m: Measurements,
) -> None:
    """ST deviation at J, J+60, J+80 ms per lead (TZ §6).

    Baseline is the PR segment (isoelectric point just before QRS onset),
    measured per beat — far more robust than a global median.
    """
    fs = sig.sampling_rate_hz
    n = min(len(qrs_on), len(qrs_off))
    onsets = np.asarray(qrs_on[:n], dtype=float)
    offsets = np.asarray(qrs_off[:n], dtype=float)
    pairs = [(int(a), int(b)) for a, b in zip(onsets, offsets)
             if np.isfinite(a) and np.isfinite(b)]
    if not pairs:
        return
    s60 = int(0.06 * fs)
    s80 = int(0.08 * fs)
    pr_off = int(0.02 * fs)  # 20 ms before QRS onset = PR-segment baseline
    st_list: list[STMeasurement] = []
    for li, lead in enumerate(sig.lead_names):
        row = sig.signal[li]
        j_vals, j60_vals, j80_vals = [], [], []
        for on, j in pairs:
            base_idx = max(0, on - pr_off)
            baseline = float(row[base_idx])
            if j < len(row):
                j_vals.append(row[j] - baseline)
            if j + s60 < len(row):
                j60_vals.append(row[j + s60] - baseline)
            if j + s80 < len(row):
                j80_vals.append(row[j + s80] - baseline)
        st_list.append(
            STMeasurement(
                lead=lead,
                j_point_uv=round(float(np.median(j_vals)), 1) if j_vals else None,
                j60_uv=round(float(np.median(j60_vals)), 1) if j60_vals else None,
                j80_uv=round(float(np.median(j80_vals)), 1) if j80_vals else None,
            )
        )
    m.morphology.st = st_list


def _measure_t_morphology(
    sig: SignalRepresentation, rpeaks: np.ndarray, fs: float, m: Measurements
) -> None:
    """Classify T-wave morphology per lead (peaked / flat / inverted) (TZ §6)."""
    if len(rpeaks) < 2:
        return
    rr = int(np.median(np.diff(rpeaks))) if len(rpeaks) >= 2 else int(0.8 * fs)
    t_start = int(0.10 * fs)
    t_end = min(int(0.45 * fs), rr - 1)
    for li, lead in enumerate(sig.lead_names):
        row = sig.signal[li]
        qrs_amp = abs(_qrs_amplitude(row, rpeaks, fs)) + 1e-6
        t_amps = []
        for r in rpeaks:
            a, b = r + t_start, r + t_end
            if b < len(row) and a < b:
                seg = row[a:b] - float(np.median(row))
                t_amps.append(seg[np.argmax(np.abs(seg))])
        if not t_amps:
            continue
        t_amp = float(np.median(t_amps))
        ratio = t_amp / qrs_amp
        if t_amp < 0 and abs(t_amp) > 100.0:
            m.morphology.inverted_t_leads.append(lead)
        elif ratio > 0.5 and t_amp > 400.0:
            m.morphology.peaked_t_leads.append(lead)
        elif abs(t_amp) < 100.0:
            m.morphology.flat_t_leads.append(lead)
