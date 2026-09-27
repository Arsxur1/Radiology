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
    STMeasurement,
)
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer3_measure.atrial import atrial_activity
from cardiolens.layer3_measure.delineation import (
    P_ONSET_MIN_FRACTION,
    P_ONSET_MIN_UV,
    qrs_bounds_envelope,
    qt_multilead,
    refine_p_onsets,
    spatial_velocity,
    terminal_qrs_sign,
)
from cardiolens.layer3_measure.intervals import (
    qtc_bazett,
    qtc_framingham,
    qtc_fridericia,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Lead preferred for rhythm/delineation.
_RHYTHM_LEAD_PREFERENCE = ["II", "V5", "V2", "I"]

# A beat's P wave counts as "present/organized" when its P-region peak-to-peak is at
# least this fraction of the R-wave amplitude reference. Below it (e.g. AF fibrillatory
# waves) the beat contributes to a low p_wave_ratio, which the AF rule keys on (§7.1).
# Calibrated on the PhysioNet ecg-arrhythmia set (sinus P/R ~0.21 vs AF ~0.10).
P_PRESENCE_FRACTION = 0.15

# PR-segment window used as the ST baseline: [onset-50 ms, onset-10 ms].
ST_BASELINE_MS = (50.0, 10.0)

# Non-QRS detection filter: envelope peak below this fraction of the 90th-percentile
# envelope peak across detections = P or T wave, not a beat.
OVERSENSE_ENV_FRACTION = 0.25

# Beat-to-beat PR standard deviation below this = a fixed PR = conducted rhythm;
# an AV-dissociation candidate with a stable PR is rejected.
PR_STD_DISSOCIATION_MS = 40.0
# ... and a dissociation candidate needs organised P waves (AF has none).
AV_DISSOCIATION_MIN_P_RATIO = 0.3


def _pick_rhythm_lead(sig: SignalRepresentation) -> int:
    for name in _RHYTHM_LEAD_PREFERENCE:
        if name in sig.lead_names:
            return sig.lead_names.index(name)
    return 0


def _reject_non_qrs_detections(signal: np.ndarray, rpeaks: np.ndarray, fs: float) -> np.ndarray:
    """Drop peak detections that are a P or T wave, not a QRS.

    At slow rates or after long pauses the adaptive peak detector accepts a tall
    T wave or a P wave as a beat. Amplitude cannot tell them apart robustly (a
    small conducted beat next to a tall PVC, a lead-II R flattened by axis
    deviation), but slope can: the multi-lead spatial-velocity envelope peaks at
    >= 0.7 of the record's reference on every true QRS and at <= 0.11 on P/T waves
    (PTB-XL and synthetic checks). A detection below OVERSENSE_ENV_FRACTION of the
    90th-percentile envelope peak is discarded.
    """
    if len(rpeaks) < 3:
        return rpeaks
    env = spatial_velocity(signal, fs)
    w = int(0.03 * fs)
    peaks = np.array([float(env[max(0, int(r) - w):int(r) + w].max()) for r in rpeaks])
    ref = float(np.percentile(peaks, 90))
    if ref <= 0:
        return rpeaks
    return rpeaks[peaks >= OVERSENSE_ENV_FRACTION * ref]


def _aligned(values, n: int) -> np.ndarray:
    """neurokit wave marks as a float array of length ``n`` (NaN-padded)."""
    arr = np.full(n, np.nan)
    vals = np.asarray(list(values)[:n], dtype=float)
    arr[:len(vals)] = vals
    return arr


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

    rpeaks = _reject_non_qrs_detections(sig.signal, rpeaks, fs)
    m.r_peaks_samples = rpeaks.tolist()
    m.n_beats = len(rpeaks)

    rr_s_mean = None
    if len(rpeaks) >= 2:
        rr = np.diff(rpeaks) / fs  # seconds
        rr_s_mean = float(np.mean(rr))
        m.heart_rate_bpm = round(60.0 / rr_s_mean, 1)
        m.rr_std_ms = round(float(np.std(rr) * 1000.0), 1)
        m.hrv_sdnn_ms = round(float(np.std(rr) * 1000.0), 1)
        # Regularity: coefficient of variation of RR < 10% -> regular.
        cv = float(np.std(rr) / (np.mean(rr) + 1e-9))
        m.rr_cv = round(cv, 3)
        m.rhythm_regular = bool(cv < 0.10)
    else:
        m.notes.append("insufficient R-peaks for rhythm analysis (low confidence)")

    # --- QRS bounds from the multi-lead envelope + dwt for P/T ---
    # The spatial-velocity envelope gives QRS onsets/offsets that hold up on real
    # BBB morphology (dwt under-estimates width); dwt still supplies P and T marks.
    n_beats = len(rpeaks)
    qrs_on = np.full(n_beats, np.nan)
    qrs_off = np.full(n_beats, np.nan)
    t_off: np.ndarray | None = None
    if n_beats >= 3:
        qrs_on, qrs_off = qrs_bounds_envelope(sig.signal, rpeaks, fs)
        p_on = p_off = None
        waves = _delineate(cleaned, rpeaks, fs)
        if waves is not None:
            p_on = _aligned(waves.get("ECG_P_Onsets", []), n_beats)
            p_off = _aligned(waves.get("ECG_P_Offsets", []), n_beats)
            dwt_on = _aligned(waves.get("ECG_R_Onsets", []), n_beats)
            dwt_off = _aligned(waves.get("ECG_R_Offsets", []), n_beats)
            t_off = _aligned(waves.get("ECG_T_Offsets", []), n_beats)
            # Fall back to dwt bounds only where the envelope found nothing.
            qrs_on = np.where(np.isfinite(qrs_on), qrs_on, dwt_on)
            qrs_off = np.where(np.isfinite(qrs_off), qrs_off, dwt_off)
        else:
            m.notes.append("delineation failed (P/T marks unavailable, low confidence)")

        base = float(np.median(cleaned))
        r_ref = float(np.percentile(np.abs(cleaned - base), 99)) or 1.0

        # PR from amplitude-refined P onsets: dwt onsets land late on real signals,
        # which under-estimates PR and hides first-degree AV block.
        p_on_ref = refine_p_onsets(
            cleaned, qrs_on, t_off, fs, max(P_ONSET_MIN_UV, P_ONSET_MIN_FRACTION * r_ref))
        pr_list = [qrs_on[i] - p_on_ref[i] for i in range(n_beats)
                   if np.isfinite(p_on_ref[i]) and np.isfinite(qrs_on[i])]
        m.p_onset_ratio = round(len(pr_list) / max(1, n_beats), 2)
        if len(pr_list) >= 3:
            m.pr_std_ms = round(float(np.std(pr_list) / fs * 1000.0), 1)
        if len(pr_list) < 0.5 * n_beats:
            pr_list = ([qrs_on[i] - p_on[i] for i in range(n_beats)
                        if np.isfinite(p_on[i]) and np.isfinite(qrs_on[i])]
                       if p_on is not None else [])
            if pr_list:
                m.notes.append("PR from dwt P onsets (refined P on <50% of beats)")
        qrs_list = [qrs_off[i] - qrs_on[i] for i in range(n_beats)
                    if np.isfinite(qrs_on[i]) and np.isfinite(qrs_off[i])
                    and qrs_off[i] > qrs_on[i]]
        # QT: tangent-method T offset per lead, measured in the lead with the longest
        # QT (guideline practice); the single-lead dwt offset is only a fallback.
        qt_tan, qt_lead, t_off_tan = qt_multilead(sig.signal, sig.lead_names, rpeaks,
                                                  qrs_on, qrs_off, fs)
        if qt_tan is not None:
            qt_ms, qt_source = qt_tan, "tangent"
            t_off = t_off_tan
        else:
            qt_list = ([t_off[i] - qrs_on[i] for i in range(n_beats)
                        if np.isfinite(t_off[i]) and np.isfinite(qrs_on[i])]
                       if t_off is not None else [])
            qt_ms, qt_source = _median_ms(qt_list, fs), ("dwt" if qt_list else None)
            if qt_ms is not None:
                m.notes.append("QT from dwt T offset (tangent method found no usable lead)")
        m.intervals = Intervals(
            pr_ms=_median_ms(pr_list, fs),
            qrs_ms=_median_ms(qrs_list, fs),
            qt_ms=qt_ms, qt_lead=qt_lead, qt_source=qt_source,
        )

        if p_on is not None and p_off is not None:
            # Fraction of beats with an *organized* P wave, by amplitude (low ->
            # suspect AF, §7.1). neurokit's delineation interpolates a P onset for
            # nearly every beat, so an onset-only ratio saturates at 1.0 even in AF
            # (where P waves are replaced by fibrillatory waves). Gate on the P-region
            # deflection instead: a peak-to-peak below ~15% of the R reference counts
            # as an absent/disorganized P.
            p_present = 0
            for i in range(n_beats):
                a, b = p_on[i], p_off[i]
                if np.isfinite(a) and np.isfinite(b) and 0 <= a < b < len(cleaned):
                    seg = cleaned[int(a):int(b)] - base
                    if seg.size and float(seg.max() - seg.min()) >= P_PRESENCE_FRACTION * r_ref:
                        p_present += 1
            m.p_wave_ratio = round(p_present / max(1, n_beats), 2)
            # P-wave duration/amplitude for atrial enlargement (P-mitrale/-pulmonale).
            pdur = [(p_off[i] - p_on[i]) for i in range(n_beats)
                    if np.isfinite(p_on[i]) and np.isfinite(p_off[i]) and p_off[i] > p_on[i]]
            m.p_wave_duration_ms = _median_ms(pdur, fs)
            m.p_wave_amplitude_uv = _p_amplitude(sig, p_on, p_off)

        # ST measured off the envelope QRS offsets (J points), PR-segment baseline.
        _measure_st(sig, rpeaks, qrs_on, qrs_off, m)

        # Atrial activity independent of the QRS: flutter / AV dissociation (§7.1).
        atrial_leads = {ln: sig.lead(ln).astype(np.float64)
                        for ln in ("II", "V1") if ln in sig.lead_names}
        if not atrial_leads:
            atrial_leads = {sig.lead_names[lead_idx]: cleaned}
        pr_unstable = bool((m.p_onset_ratio or 0.0) >= 0.5 and m.pr_std_ms is not None
                           and m.pr_std_ms >= PR_STD_DISSOCIATION_MS)
        rate, av, notes = atrial_activity(
            atrial_leads, rpeaks, fs,
            heart_rate_bpm=m.heart_rate_bpm, rr_cv=m.rr_cv, r_ref=r_ref,
            pr_unstable=pr_unstable, qrs_on=qrs_on, qrs_off=qrs_off,
        )
        # AV dissociation means no fixed PR: when a P onset was found before most
        # beats and the PR is stable, the atria are conducting (sinus rhythm with a
        # long PR / long QT left a periodic residual) — not a complete block.
        if (av is True and (m.p_onset_ratio or 0.0) >= 0.5
                and m.pr_std_ms is not None and m.pr_std_ms < PR_STD_DISSOCIATION_MS):
            av = False
            notes.append(f"atrial activity: stable PR (std {m.pr_std_ms} ms) -> conducted, not dissociated")
        # A complete block has P waves by definition: without organised P waves the
        # periodic residual is fibrillatory/noise (AF with a slow response), not
        # dissociation.
        if (av is True and (m.p_wave_ratio or 0.0) < AV_DISSOCIATION_MIN_P_RATIO
                and (m.p_onset_ratio or 0.0) < 0.5):
            av = False
            notes.append("atrial activity: no organised P waves -> not AV dissociation")
        m.atrial_rate_bpm, m.av_dissociation = rate, av
        m.notes.extend(notes)

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

    # --- Terminal QRS forces in V1/V6 (RSR' / broad-S typing for BBB) ---
    for lead, attr in (("V1", "qrs_v1_terminal_positive"), ("V6", "qrs_v6_terminal_positive")):
        if lead in sig.lead_names:
            setattr(m, attr, terminal_qrs_sign(sig.lead(lead), rpeaks, qrs_off, fs))

    # --- Limb-lead QRS amplitude (for low-voltage detection, TZ §7.5) ---
    m.qrs_amplitude_limb_uv = _limb_qrs_amplitude(sig, rpeaks, fs)

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


def _p_amplitude(sig: SignalRepresentation, p_on: np.ndarray, p_off: np.ndarray) -> float | None:
    """Median P-wave amplitude in lead II (µV) — P-pulmonale if tall."""
    if "II" not in sig.lead_names:
        return None
    row = sig.lead("II")
    baseline = float(np.median(row))
    amps = []
    n = min(len(p_on), len(p_off))
    for i in range(n):
        a, b = p_on[i], p_off[i]
        if np.isfinite(a) and np.isfinite(b) and 0 <= a < b < len(row):
            seg = row[int(a):int(b)] - baseline
            if seg.size:
                amps.append(float(np.max(seg)))
    return round(float(np.median(amps)), 1) if amps else None


def _limb_qrs_amplitude(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    """Median QRS peak-to-peak across the limb leads I/II/III (µV, TZ §7.5)."""
    limb = [ln for ln in ("I", "II", "III") if ln in sig.lead_names]
    if not limb or len(rpeaks) == 0:
        return None
    win = int(0.05 * fs)
    amps = []
    for lead in limb:
        row = sig.lead(lead)
        for r in rpeaks:
            a, b = max(0, r - win), min(len(row), r + win)
            seg = row[a:b]
            amps.append(float(np.max(seg) - np.min(seg)))
    return round(float(np.median(amps)), 1) if amps else None


def _measure_st(
    sig: SignalRepresentation,
    rpeaks: np.ndarray,
    qrs_on: np.ndarray,
    qrs_off: np.ndarray,
    m: Measurements,
) -> None:
    """ST deviation at J, J+60, J+80 ms per lead (TZ §6).

    J = envelope QRS offset. Baseline = median of the PR segment
    ([onset-50, onset-10] ms) per beat — a single sample is too noisy on real
    recordings and produced spurious ST calls. Also records the net QRS
    deflection per lead so Layer 4 can judge concordance (secondary ST-T changes).
    """
    fs = sig.sampling_rate_hz
    pairs = [(int(a), int(b)) for a, b in zip(qrs_on, qrs_off)
             if np.isfinite(a) and np.isfinite(b) and b > a]
    if not pairs:
        return
    s60, s80 = int(0.06 * fs), int(0.08 * fs)
    b_lo, b_hi = int(ST_BASELINE_MS[0] / 1000.0 * fs), int(ST_BASELINE_MS[1] / 1000.0 * fs)
    st_list: list[STMeasurement] = []
    for li, lead in enumerate(sig.lead_names):
        row = sig.signal[li]
        j_vals, j60_vals, j80_vals = [], [], []
        for on, j in pairs:
            lo, hi = max(0, on - b_lo), max(1, on - b_hi)
            baseline = float(np.median(row[lo:hi])) if hi > lo else float(row[max(0, on - 1)])
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
                qrs_net_uv=round(_qrs_amplitude(row, rpeaks, fs), 1),
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
