"""Structured measurements produced by Layer 3 (TZ §6).

These are the quantitative features that Layer 4 (findings) and Layer 5 (causal
analysis) consume. All durations in milliseconds, amplitudes in microvolts,
angles in degrees, rate in bpm.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Intervals(BaseModel):
    """Global interval measurements (TZ §6)."""

    pr_ms: float | None = None
    qrs_ms: float | None = None
    qt_ms: float | None = None
    # Rate-corrected QT with three formulas (TZ §6).
    qtc_bazett_ms: float | None = None
    qtc_fridericia_ms: float | None = None
    qtc_framingham_ms: float | None = None


class Axes(BaseModel):
    """Electrical axes (TZ §6)."""

    p_deg: float | None = None
    qrs_deg: float | None = None
    t_deg: float | None = None


class STMeasurement(BaseModel):
    """ST-segment deviation per lead at J and J+60/80 ms (TZ §6)."""

    lead: str
    j_point_uv: float | None = None
    j60_uv: float | None = None
    j80_uv: float | None = None


class Morphology(BaseModel):
    """Waveform morphology descriptors (TZ §6)."""

    st: list[STMeasurement] = Field(default_factory=list)
    # Leads exhibiting each morphology, for evidence trails.
    peaked_t_leads: list[str] = Field(default_factory=list)
    flat_t_leads: list[str] = Field(default_factory=list)
    inverted_t_leads: list[str] = Field(default_factory=list)
    u_wave_leads: list[str] = Field(default_factory=list)
    pathological_q_leads: list[str] = Field(default_factory=list)


class Measurements(BaseModel):
    """Full Layer-3 output (TZ §6)."""

    heart_rate_bpm: float | None = None
    rhythm_regular: bool | None = None
    rr_std_ms: float | None = None            # RR interval variability
    rr_cv: float | None = None                # RR coefficient of variation
    rr_mad_cv: float | None = None            # robust RR dispersion (1.4826·MAD / median)
    hrv_sdnn_ms: float | None = None
    max_rr_ms: float | None = None            # longest RR gap (pause detection, §7.1/§10)
    pause_ratio: float | None = None          # longest RR / median RR (dropped-beat cue)
    # Calibrated PR per lead — multi-lead PR agreement improves first-degree AV
    # block detection (a single lead under-measures PR; §7.3).
    pr_ms_by_lead: dict = Field(default_factory=dict)
    # Per-lead QT, informational (D28); the reported QT is the rhythm lead's.
    qt_ms_by_lead: dict = Field(default_factory=dict)
    p_wave_ratio: float | None = None         # fraction of beats with a detected P
    qrs_v1_terminal_positive: bool | None = None  # RSR' sign for BBB typing
    qrs_v1_rs_ratio: float | None = None          # R/S in V1 (dominant R -> RVH, §7.5)
    qrs_amplitude_limb_uv: float | None = None    # median limb-lead QRS peak-to-peak
    # NOT YET MEASURED (always None): Layer 3 has no robust estimator for these,
    # so the atrial-flutter and complete-AV-block rules that consume them are
    # currently inert. Flutter needs ventricular-QRS cancellation + atrial-
    # organisation analysis (a single-lead FFT peak in the flutter band false-
    # positives on ~1/4 of non-flutter records — validated on ecg-arrhythmia,
    # n=67); complete AV block needs P-to-QRS association. Both are v2 (see
    # DECISIONS.md D15a). Leaving them None keeps the rules from firing on
    # fabricated inputs (§0).
    atrial_rate_bpm: float | None = None          # P/flutter-wave rate (TZ §7.1)
    av_dissociation: bool | None = None           # complete AV block signal (TZ §7.1)
    p_wave_duration_ms: float | None = None       # atrial enlargement (P-mitrale)
    p_wave_amplitude_uv: float | None = None      # atrial enlargement (P-pulmonale)
    n_beats: int = 0
    r_peaks_samples: list[int] = Field(default_factory=list)
    intervals: Intervals = Field(default_factory=Intervals)
    axes: Axes = Field(default_factory=Axes)
    morphology: Morphology = Field(default_factory=Morphology)
    # Sokolow-Lyon voltage sum for LVH assessment (µV).
    sokolow_lyon_uv: float | None = None
    # Beat-template consistency in [0,1]: median correlation of each QRS complex
    # to the median beat. High for real ECG (incl. AFib); low for artifact/noise
    # where "R-peaks" are spurious — a signal-organization quality gate (§5).
    qrs_template_consistency: float | None = None
    notes: list[str] = Field(default_factory=list)  # TODO / low-confidence markers
