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
from cardiolens.layer3_measure.global_beat import (
    global_initial_qrs_ratio,
    global_intervals_ex,
    global_q_waves,
    global_qrs_amplitudes,
    global_qrs_rs,
    global_right_precordial_st,
    global_st,
    global_t_amplitude,
    global_terminal_s,
)
from cardiolens.layer3_measure.intervals import (
    qtc_bazett,
    qtc_framingham,
    qtc_fridericia,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Lead preferred for rhythm/delineation.
_RHYTHM_LEAD_PREFERENCE = ["II", "V5", "V2", "I"]

# Slope-energy QRS-width estimator: fraction of the local envelope peak that
# bounds the contiguous complex. Calibrated on LUDB expert gold (D15o): 0.05
# gives bias +4 ms / MAE 14 (fiducial: -5 / 12) yet separates normals
# (median 86 ms, 1% >=120) from complete blocks (CRBBB 136/89%, CLBBB 153/100%),
# where the fiducial offset over-reads normals (40% >=120) and under-reads RBBB.
_QRS_ENERGY_FRACTION = 0.05


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


def _pick_rhythm_lead(sig: SignalRepresentation) -> int:
    for name in _RHYTHM_LEAD_PREFERENCE:
        if name in sig.lead_names:
            return sig.lead_names.index(name)
    return 0


def _median_ms(
    values: list[float], fs: float, *, lo_ms: float | None = None, hi_ms: float | None = None
) -> float | None:
    """Median of per-beat interval samples, in ms.

    Beat-level delineation on real records throws physiologically impossible
    values — a P onset landing just before the QRS yields a "PR" of ~70 ms, a
    missed T offset a negative QT. ``lo_ms``/``hi_ms`` drop per-beat values
    outside a plausible window before the median, so a handful of bad fiducials
    on one beat can't drag the whole record's interval down (or up).
    """
    vals = [v / fs * 1000.0 for v in values if v is not None and np.isfinite(v)]
    if lo_ms is not None:
        vals = [v for v in vals if v >= lo_ms]
    if hi_ms is not None:
        vals = [v for v in vals if v <= hi_ms]
    if not vals:
        return None
    return float(np.median(vals))


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


def _apply_correction(value: float | None, offset: float) -> float | None:
    """Re-centre a measured interval by the calibrated fiducial-bias offset (ms)."""
    if value is None or not offset:
        return value
    return round(value + offset, 1)


def _intervals_on_lead(lead: np.ndarray, rpeaks: np.ndarray, fs: float) -> tuple[float | None, float | None]:
    """Median PR and QT (ms, raw) on one lead, reusing shared R-peaks (§7.3)."""
    waves = _delineate(lead, rpeaks, fs)
    if waves is None:
        return None, None
    p_on = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
    qrs_on = np.asarray(waves.get("ECG_R_Onsets", []), dtype=float)
    t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)
    n = min(len(p_on), len(qrs_on))
    pr = [qrs_on[i] - p_on[i] for i in range(n)
          if np.isfinite(p_on[i]) and np.isfinite(qrs_on[i])]
    k = min(len(qrs_on), len(t_off))
    qt = [t_off[i] - qrs_on[i] for i in range(k)
          if np.isfinite(qrs_on[i]) and np.isfinite(t_off[i])]
    return (_median_ms(pr, fs, lo_ms=80, hi_ms=600),
            _median_ms(qt, fs, lo_ms=200, hi_ms=900))


def _pr_on_lead(lead: np.ndarray, rpeaks: np.ndarray, fs: float) -> float | None:
    """Median PR (ms, raw) on one lead (§7.3 multi-lead PR)."""
    return _intervals_on_lead(lead, rpeaks, fs)[0]


# Leads with an informational per-lead QT (D28); the rhythm lead's QT is added.
# Only leads already delineated for PR: an extra delineation per record just
# for an informational value cost ~20% of measure() time.
_QT_LEADS = ("V1", "V2", "aVF")
_PR_LEADS = ("V1", "V2", "aVF")


def measure(sig: SignalRepresentation, *, correction=None) -> Measurements:
    """Full Layer-3 measurement pass (TZ §6).

    ``correction`` is an optional object with ``pr_ms``/``qrs_ms``/``qt_ms`` offsets
    (a ``DelineationCorrection``) that re-centre the measured intervals against the
    delineator's systematic fiducial bias (DECISIONS.md D15a). None = no correction.
    """
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
        # Robust dispersion (MAD / median, normal-consistent): sinus rhythm with a
        # few ectopic beats keeps most RR clustered at the median, so a couple of
        # coupling/compensatory outliers inflate the std-based CV but not this.
        if len(rr) >= 4:
            med = float(np.median(rr))
            m.rr_mad_cv = round(1.4826 * float(np.median(np.abs(rr - med))) / (med + 1e-9), 3)
        # Longest RR gap and its ratio to the median beat — pause / dropped-beat
        # detection (§7.1, §10). Purely R-peak based, so as reliable as HR.
        m.max_rr_ms = round(float(np.max(rr)) * 1000.0, 1)
        m.pause_ratio = round(float(np.max(rr) / (np.median(rr) + 1e-9)), 2)
    else:
        m.notes.append("insufficient R-peaks for rhythm analysis (low confidence)")

    # --- Delineation -> intervals ---
    rs_median: dict[str, tuple[float, float]] = {}
    if len(rpeaks) >= 3:
        waves = _delineate(cleaned, rpeaks, fs)
        if waves is not None:
            p_on = np.asarray(waves.get("ECG_P_Onsets", []), dtype=float)
            p_off = np.asarray(waves.get("ECG_P_Offsets", []), dtype=float)
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

            # Reject per-beat delineation outliers before the median. The lower
            # bounds catch the common failure (a P onset just before the QRS -> a
            # "PR" of ~70 ms; a missed T offset -> a negative QT). The UPPER bounds
            # are deliberately supraphysiologic, not "normal": they must exclude
            # only impossible spans (a QT that ran into the next beat) while
            # preserving the extreme-but-real pathology the safety layer exists to
            # catch — a 720 ms long QT, a 220 ms hyperkalaemic/paced QRS, a 430 ms
            # PR in high-grade AV block. Capping at normal limits would null the
            # interval and silently suppress exactly those findings.
            pr_ms = _median_ms(pr_list, fs, lo_ms=80, hi_ms=600)
            qrs_ms = _median_ms(qrs_list, fs, lo_ms=40, hi_ms=300)
            qt_ms = _median_ms(qt_list, fs, lo_ms=200, hi_ms=900)
            # Re-centre against the delineator's measured fiducial bias (D15a).
            if correction is not None:
                pr_ms = _apply_correction(pr_ms, getattr(correction, "pr_ms", 0.0))
                qrs_ms = _apply_correction(qrs_ms, getattr(correction, "qrs_ms", 0.0))
                qt_ms = _apply_correction(qt_ms, getattr(correction, "qt_ms", 0.0))
            # Prefer the robust slope-energy QRS width (D15o): the fiducial QRS
            # offset is unstable in BOTH directions — it under-reads wide blocks
            # (RBBB) and over-reads normals (false LBBB). The energy width, gold-
            # calibrated on LUDB, fixes both. Fiducial value stands only if the
            # envelope is not computable. The PR fiducials / QRS onsets used for
            # ST and PR are left untouched.
            qrs_energy = _qrs_width_energy(sig, rpeaks, fs)
            if qrs_energy is not None:
                qrs_ms = qrs_energy
            m.intervals = Intervals(pr_ms=pr_ms, qrs_ms=qrs_ms, qt_ms=qt_ms)
            # Global PR and QT from the 12-lead median beat (D30) replace the noisy
            # rhythm-lead values when they can be measured; the per-lead values are kept.
            g_pr, g_qt, t_found = global_intervals_ex(sig.signal, rpeaks, fs)
            if g_pr is not None:
                m.pr_ms_fiducial = pr_ms
                gp_off = getattr(correction, "global_pr_ms", 0.0) if correction is not None else 0.0
                m.intervals.pr_ms = round(g_pr + gp_off, 1)
                m.pr_global = True
            if g_qt is not None:
                m.qt_ms_fiducial = qt_ms
                g_off = getattr(correction, "global_qt_ms", 0.0) if correction is not None else 0.0
                m.intervals.qt_ms = round(g_qt + g_off, 1)
            elif t_found is False:
                # Above 100/min the T end ran onto the next beat's limit: a QT read there is
                # the limit, not the T wave (infants, qtdb). The single-lead value in those
                # windows was off by up to +390 ms, so QT stays unmeasured (D102).
                m.qt_ms_fiducial = qt_ms
                m.intervals.qt_ms = None
                m.qt_unmeasurable = True
            # Multi-lead PR (§7.3): a single lead under-measures PR, so first-degree
            # AV block is called on agreement across leads (layer 4). Reuse the
            # already-detected R-peaks; only PR is recomputed per lead.
            pr_by_lead: dict[str, float] = {}
            if pr_ms is not None:
                pr_by_lead["II"] = pr_ms
            pr_corr = getattr(correction, "pr_ms", 0.0) if correction is not None else 0.0
            qt_corr = getattr(correction, "qt_ms", 0.0) if correction is not None else 0.0
            qt_by_lead: dict[str, float] = {}
            if qt_ms is not None:
                qt_by_lead[sig.lead_names[lead_idx]] = qt_ms
            for lead_name in dict.fromkeys(_PR_LEADS + _QT_LEADS):
                if lead_name not in sig.lead_names or lead_name == sig.lead_names[lead_idx]:
                    continue
                lead_pr, lead_qt = _intervals_on_lead(
                    sig.signal[sig.lead_names.index(lead_name)].astype(np.float64), rpeaks, fs)
                if lead_pr is not None and lead_name in _PR_LEADS:
                    pr_by_lead[lead_name] = round(lead_pr + pr_corr, 1)
                if lead_qt is not None and lead_name in _QT_LEADS:
                    qt_by_lead[lead_name] = round(lead_qt + qt_corr, 1)
            m.pr_ms_by_lead = pr_by_lead
            # Per-lead QT is informational only (D28): a multi-lead median as the
            # reported QT lost long-QT sensitivity on held-out data, so the
            # rhythm-lead QT stays the reported value.
            m.qt_ms_by_lead = qt_by_lead
            # Initial Q waves per lead on the median beat (pathological Q, D36).
            m.morphology.q_waves = global_q_waves(sig.signal, rpeaks, fs, list(sig.lead_names))
            # QRS size per lead on the median beat: low voltage (D61).
            amps = global_qrs_amplitudes(sig.signal, rpeaks, fs, list(sig.lead_names))
            limb = [amps[k] for k in ("I", "II", "III", "aVR", "aVL", "aVF") if k in amps]
            prec = [amps[k] for k in ("V1", "V2", "V3", "V4", "V5", "V6") if k in amps]
            m.qrs_amplitude_limb_max_uv = max(limb) if len(limb) == 6 else None
            m.qrs_amplitude_precordial_max_uv = max(prec) if len(prec) == 6 else None
            # Slurred QRS upstroke (delta wave) on the median beat (WPW, D59).
            m.qrs_initial_energy_ratio = global_initial_qrs_ratio(sig.signal, rpeaks, fs)
            # ST on the median beat (D48) — decides the critical STEMI tier.
            m.morphology.st_median_beat = global_st(sig.signal, rpeaks, fs, list(sig.lead_names))
            # T peak on the median beat (D49) — decides the critical hyperkalaemia tier.
            m.morphology.t_median_beat = global_t_amplitude(sig.signal, rpeaks, fs,
                                                            list(sig.lead_names))
            # R/S per lead on the median beat: largest deflection (peaked T, D65)
            # and Sokolow-Lyon (D66, set below).
            rs_median = global_qrs_rs(sig.signal, rpeaks, fs, list(sig.lead_names))
            m.morphology.qrs_extreme_median_beat = {k: max(r, -s)
                                                    for k, (r, s) in rs_median.items()}
            m.morphology.qrs_rs_median_beat = rs_median
            # Right precordial ST shape (Brugada type 1, D89).
            m.morphology.right_precordial_st = global_right_precordial_st(
                sig.signal, rpeaks, fs, list(sig.lead_names))
            m.morphology.terminal_s_lateral = global_terminal_s(
                sig.signal, rpeaks, fs, list(sig.lead_names))
            # Fraction of beats with a detected P wave (low -> suspect AF, §7.1).
            p_valid = int(np.sum(np.isfinite(p_on)))
            m.p_wave_ratio = round(p_valid / max(1, len(rpeaks)), 2)
            # P-wave duration/amplitude for atrial enlargement (P-mitrale/-pulmonale).
            pdur = [(p_off[i] - p_on[i]) for i in range(min(len(p_on), len(p_off)))
                    if np.isfinite(p_on[i]) and np.isfinite(p_off[i]) and p_off[i] > p_on[i]]
            m.p_wave_duration_ms = _median_ms(pdur, fs, lo_ms=40, hi_ms=200)
            m.p_wave_amplitude_uv = _p_amplitude(sig, p_on, p_off)
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

    # --- Frontal QRS axis: median-beat net R + S in I and aVF (D69), else per beat ---
    axis = _qrs_axis_median(rs_median)
    m.axes = Axes(qrs_deg=axis if axis is not None else _qrs_axis(sig, rpeaks, fs))

    # --- Beat-template consistency (signal-organization quality gate, §5) ---
    m.qrs_template_consistency = _qrs_template_consistency(sig, rpeaks, fs)

    # --- Sokolow-Lyon voltage (LVH) ---
    # Sokolow-Lyon from the median beat when available (D66), else per beat.
    m.sokolow_lyon_uv = _sokolow_lyon_median(rs_median)
    if m.sokolow_lyon_uv is None:
        m.sokolow_lyon_uv = _sokolow_lyon(sig, rpeaks, fs)

    # --- V1 terminal QRS sign (RSR' typing for bundle branch block) ---
    m.qrs_v1_terminal_positive = _v1_terminal_positive(sig, rpeaks, fs)

    # --- V1 R/S ratio (dominant R in V1 -> RVH, posterior MI; TZ §7.5) ---
    m.qrs_v1_rs_ratio = _v1_rs_ratio(sig, rpeaks, fs)

    # --- Limb-lead QRS amplitude (for low-voltage detection, TZ §7.5) ---
    m.qrs_amplitude_limb_uv = _limb_qrs_amplitude(sig, rpeaks, fs)

    # --- Morphology (peaked/flat/inverted T) ---
    m.morphology.st = m.morphology.st  # populated by _measure_st
    _measure_t_morphology(sig, rpeaks, fs, m)

    return m


def _qrs_template_consistency(
    sig: SignalRepresentation, rpeaks: np.ndarray, fs: float,
) -> float | None:
    """Median correlation of each QRS complex to the median beat, in [0,1] (§5).

    Real ECG has a self-consistent QRS shape from beat to beat — even in atrial
    fibrillation, where timing is irregular but morphology is stable — so this
    stays high (~0.98). Band-limited noise and artifact have no repeatable beat,
    so spurious "R-peaks" correlate poorly (~0.6-0.75). It is the organization
    signal a plain SNR/HF quality index misses once the band-pass has smoothed
    broadband noise into the ECG band. ``None`` if too few beats to judge.
    """
    if len(rpeaks) < 3:
        return None
    li = sig.lead_names.index("II") if "II" in sig.lead_names else 0
    row = sig.signal[li].astype(np.float64)
    # +-90 ms captures the full QRS even for wide (BBB/paced) complexes, and the
    # wider window also drives noise correlation down (a random signal decorrelates
    # over a longer span) — sharpening the separation from real ECG.
    w = int(0.09 * fs)
    beats = [row[int(r) - w:int(r) + w] for r in rpeaks
             if int(r) - w >= 0 and int(r) + w < len(row)]
    if len(beats) < 3:
        return None
    m = np.vstack(beats)
    templ = np.median(m, axis=0)
    if np.std(templ) < 1e-9:
        return 0.0
    cors = [float(np.corrcoef(b, templ)[0, 1]) if np.std(b) > 1e-9 else 0.0 for b in m]
    return round(float(np.median(cors)), 3)


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


def _v1_rs_ratio(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    """Median R/S amplitude ratio in V1 (dominant R -> RVH / posterior MI, §7.5)."""
    if "V1" not in sig.lead_names or len(rpeaks) == 0:
        return None
    row = sig.lead("V1").astype(float)
    base = float(np.median(row))
    win = int(0.05 * fs)
    ratios = []
    for r in rpeaks:
        a, b = max(0, r - win), min(len(row), r + win)
        seg = row[a:b] - base
        if seg.size == 0:
            continue
        R = max(float(seg.max()), 1e-6)   # tallest positive (R)
        S = max(float(-seg.min()), 1e-6)  # deepest negative (S)
        ratios.append(R / S)
    return round(float(np.median(ratios)), 2) if ratios else None


def _qrs_axis_median(rs: dict[str, tuple[float, float]]) -> float | None:
    """Frontal QRS axis from the net deflection (R + S) in I and aVF on the median beat
    (D69). Against GE 12SL: MAE 6.0 deg (per-beat signed extreme: 8.3 deg)."""
    if not {"I", "aVF"}.issubset(rs):
        return None
    net_i, net_f = sum(rs["I"]), sum(rs["aVF"])
    if net_i == 0.0 and net_f == 0.0:
        return None
    return round(float(np.degrees(np.arctan2(net_f, net_i))), 1)


def _qrs_axis(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> float | None:
    if not {"I", "aVF"}.issubset(set(sig.lead_names)) or len(rpeaks) == 0:
        return None
    i_amp = _qrs_amplitude(sig.lead("I"), rpeaks, fs)
    f_amp = _qrs_amplitude(sig.lead("aVF"), rpeaks, fs)
    return round(float(np.degrees(np.arctan2(f_amp, i_amp))), 1)


def _qrs_width_energy(
    sig: SignalRepresentation, rpeaks: np.ndarray, fs: float,
    frac: float = _QRS_ENERGY_FRACTION,
) -> float | None:
    """Robust QRS duration (ms) from a multi-lead slope-energy envelope (D15o).

    The QRS is the burst of high slew shared across leads, so sum ``|derivative|``
    over all leads, smooth (~12 ms), and take the span around each R peak where
    the envelope exceeds ``frac`` of its local peak; median across beats. This
    avoids the per-lead fiducial QRS-offset instability that under-reads wide
    complexes (RBBB) and over-reads normals (false LBBB). ``None`` if not
    computable. Uses a +-110 ms search window to admit genuinely wide blocks.
    """
    if len(rpeaks) == 0:
        return None
    x = np.asarray(sig.signal, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 5:
        return None
    env = np.abs(np.diff(x, axis=1)).sum(axis=0)
    k = max(1, int(0.012 * fs))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    half = int(0.11 * fs)
    widths: list[float] = []
    for r in rpeaks:
        a, b = max(0, int(r) - half), min(len(env), int(r) + half)
        seg = env[a:b]
        if seg.size < 5:
            continue
        pk = float(seg.max())
        if pk <= 0:
            continue
        mask = seg >= frac * pk
        # Take the CONTIGUOUS run around the QRS peak, not first-to-last of all
        # above-threshold samples — otherwise a nearby T/P shoulder that pokes
        # above the threshold balloons the span across the gap between complexes.
        c = int(np.argmax(seg))
        lo = c
        while lo > 0 and mask[lo - 1]:
            lo -= 1
        hi = c
        while hi < len(mask) - 1 and mask[hi + 1]:
            hi += 1
        widths.append((hi - lo) / fs * 1000.0)
    return round(float(np.median(widths)), 1) if widths else None


def _sokolow_lyon_median(rs: dict[str, tuple[float, float]]) -> float | None:
    """Sokolow-Lyon |S V1| + max(R V5, R V6) from median-beat R/S amplitudes (D66)."""
    if not {"V1", "V5", "V6"}.issubset(rs):
        return None
    return round(-rs["V1"][1] + max(rs["V5"][0], rs["V6"][0]), 1)


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


def _v1_terminal_positive(sig: SignalRepresentation, rpeaks: np.ndarray, fs: float) -> bool | None:
    """Sign of the terminal QRS deflection in V1 (~40 ms after R).

    Positive terminal force (R') -> RBBB pattern; negative -> LBBB pattern.
    """
    if "V1" not in sig.lead_names or len(rpeaks) == 0:
        return None
    row = sig.lead("V1")
    baseline = float(np.median(row))
    off = int(0.04 * fs)
    win = int(0.02 * fs)
    vals = []
    for r in rpeaks:
        a, b = r + off - win, r + off + win
        if 0 <= a and b < len(row):
            seg = row[a:b] - baseline
            vals.append(seg[np.argmax(np.abs(seg))])
    if not vals:
        return None
    return bool(np.median(vals) > 0)


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
