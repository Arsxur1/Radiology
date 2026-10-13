"""QTc correction formulas (TZ §6)."""

from __future__ import annotations


def qtc_bazett(qt_ms: float, rr_s: float) -> float:
    return qt_ms / (rr_s ** 0.5)


def qtc_fridericia(qt_ms: float, rr_s: float) -> float:
    return qt_ms / (rr_s ** (1.0 / 3.0))


def qtc_framingham(qt_ms: float, rr_s: float) -> float:
    # Linear correction: QTc = QT + 154 * (1 - RR)  [ms, RR in s].
    return qt_ms + 154.0 * (1.0 - rr_s)
