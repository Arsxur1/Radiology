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
    qt_lead: str | None = None           # lead the QT was measured in (longest QT)
    qt_source: str | None = None         # "tangent" (multi-lead) | "dwt" (fallback)


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
    qrs_net_uv: float | None = None   # signed net QRS deflection (concordance checks)
    s_depth_uv: float | None = None   # depth of the S wave below the PR baseline (proportional ST rules)
    r_height_uv: float | None = None  # height of the R wave above the PR baseline


class Voltages(BaseModel):
    """Per-lead QRS voltages for hypertrophy criteria (µV / ms; TZ §7.4)."""

    r_avl_uv: float | None = None
    s_v3_uv: float | None = None
    r_v5_uv: float | None = None
    r_v6_uv: float | None = None
    s_v1_uv: float | None = None
    s_v2_uv: float | None = None
    r_v1_uv: float | None = None
    s_v6_uv: float | None = None
    r_max_limb_uv: float | None = None          # tallest R or deepest S in any limb lead
    intrinsicoid_v5_v6_ms: float | None = None  # QRS onset -> R peak (V5/V6, the later one)
    p_terminal_v1_uv_ms: float | None = None    # negative terminal P force in V1 (|µV| x ms)


class Morphology(BaseModel):
    """Waveform morphology descriptors (TZ §6)."""

    st: list[STMeasurement] = Field(default_factory=list)
    # Leads exhibiting each morphology, for evidence trails.
    peaked_t_leads: list[str] = Field(default_factory=list)
    flat_t_leads: list[str] = Field(default_factory=list)
    inverted_t_leads: list[str] = Field(default_factory=list)
    u_wave_leads: list[str] = Field(default_factory=list)
    pathological_q_leads: list[str] = Field(default_factory=list)
    # Delta-wave descriptor (D51): per lead, the time from QRS onset until the
    # slope first reaches half of that lead's maximal QRS slope (ms); a slurred
    # pre-excited upstroke is slow for 40-100 ms, a normal QRS for 15-35 ms.
    delta_ms_by_lead: dict[str, float] = Field(default_factory=dict)
    delta_ms: float | None = None         # median over the measurable leads


class Measurements(BaseModel):
    """Full Layer-3 output (TZ §6)."""

    heart_rate_bpm: float | None = None
    rhythm_regular: bool | None = None
    rr_std_ms: float | None = None            # RR interval variability
    rr_cv: float | None = None                # RR coefficient of variation
    hrv_sdnn_ms: float | None = None
    p_wave_ratio: float | None = None         # fraction of beats with an organised P (AF gate)
    p_onset_ratio: float | None = None        # fraction of beats with a refined P onset (PR)
    pr_std_ms: float | None = None            # beat-to-beat PR variability (AV dissociation gate)
    qrs_v1_terminal_positive: bool | None = None  # terminal forces in V1 (R' -> RBBB)
    qrs_v6_terminal_positive: bool | None = None  # terminal forces in V6 (broad R -> LBBB)
    qrs_amplitude_limb_uv: float | None = None    # median limb-lead QRS peak-to-peak
    atrial_rate_bpm: float | None = None          # P/flutter-wave rate (TZ §7.1)
    av_dissociation: bool | None = None           # complete AV block signal (TZ §7.1)
    p_wave_duration_ms: float | None = None       # atrial enlargement (P-mitrale)
    p_wave_amplitude_uv: float | None = None      # atrial enlargement (P-pulmonale)
    ml_probabilities: dict[str, float] | None = None   # learnable layer, per finding code
    n_beats: int = 0
    r_peaks_samples: list[int] = Field(default_factory=list)
    intervals: Intervals = Field(default_factory=Intervals)
    voltages: Voltages = Field(default_factory=Voltages)
    axes: Axes = Field(default_factory=Axes)
    morphology: Morphology = Field(default_factory=Morphology)
    # Sokolow-Lyon voltage sum for LVH assessment (µV).
    sokolow_lyon_uv: float | None = None
    notes: list[str] = Field(default_factory=list)  # TODO / low-confidence markers
