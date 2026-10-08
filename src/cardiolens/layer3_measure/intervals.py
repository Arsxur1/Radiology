"""QTc correction formulas (TZ §6)."""

from __future__ import annotations


def qtc_bazett(qt_ms: float, rr_s: float) -> float:
    return qt_ms / (rr_s ** 0.5)


def qtc_fridericia(qt_ms: float, rr_s: float) -> float:
    return qt_ms / (rr_s ** (1.0 / 3.0))


def qtc_framingham(qt_ms: float, rr_s: float) -> float:
    # Linear correction: QTc = QT + 154 * (1 - RR)  [ms, RR in s].
    return qt_ms + 154.0 * (1.0 - rr_s)


# Bazett over-corrects at fast rates (QTc inflates as RR shortens); AHA/ACCF/HRS 2009
# advises against it outside ~60-90/min. Above this rate the §10 long-QT alert tier
# (QTc >= 500) reads Fridericia instead; the long-QT finding keeps Bazett (D96).
# Guideline constant, not tuned.
FRIDERICIA_ABOVE_BPM = 90.0


def qtc_for_rules(intervals, heart_rate_bpm: float | None) -> tuple[float | None, str]:
    """QTc for the long-QT alert tier: Bazett, or Fridericia above 90/min (D96)."""
    if (heart_rate_bpm is not None and heart_rate_bpm > FRIDERICIA_ABOVE_BPM
            and intervals.qtc_fridericia_ms is not None):
        return intervals.qtc_fridericia_ms, "fridericia"
    return intervals.qtc_bazett_ms, "bazett"
