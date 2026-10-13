"""Serial comparison against a prior ECG (TZ §8.4).

Temporal dynamics are a strong clinical signal: a *new* ST elevation or a *new*
QT prolongation carries far more weight than the same finding in isolation
(TZ §8.4). This module diffs the current analysis against a prior one and emits
serial deltas that raise confidence and criticality and feed the Layer-6
narrative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import FindingSet

Direction = Literal["new", "resolved", "changed"]

# Newly-appeared findings that are especially significant on serial comparison.
_SIGNIFICANT_NEW = {
    "st_elevation": ("Новая элевация ST по сравнению с прошлой ЭКГ",
                     "Oldingi EKGga nisbatan yangi ST elevatsiyasi", True),
    "long_qt": ("Новое удлинение QT", "Yangi QT uzayishi", True),
    "atrial_fibrillation": ("Новая фибрилляция предсердий",
                            "Yangi bo'lmachalar fibrillyatsiyasi", False),
    "lbbb": ("Новая блокада левой ножки", "Yangi chap oyoqcha blokadasi", False),
    "rbbb": ("Новая блокада правой ножки", "Yangi o'ng oyoqcha blokadasi", False),
}

# Clinically meaningful QTc change threshold (ms).
_QTC_SIGNIFICANT_MS = 60.0


@dataclass
class SerialDelta:
    code: str
    direction: Direction
    label_ru: str
    label_uz: str
    critical: bool = False


@dataclass
class SerialComparison:
    prior_id: str | None = None
    deltas: list[SerialDelta] = field(default_factory=list)
    qtc_delta_ms: float | None = None
    hr_delta_bpm: float | None = None

    @property
    def has_significant_change(self) -> bool:
        return any(d.critical for d in self.deltas) or bool(self.deltas)

    def to_dict(self) -> dict:
        return {
            "prior_id": self.prior_id,
            "qtc_delta_ms": self.qtc_delta_ms,
            "hr_delta_bpm": self.hr_delta_bpm,
            "deltas": [
                {"code": d.code, "direction": d.direction, "label_ru": d.label_ru,
                 "label_uz": d.label_uz, "critical": d.critical}
                for d in self.deltas
            ],
        }


def compare(
    current_findings: FindingSet,
    current_m: Measurements,
    prior_findings: FindingSet,
    prior_m: Measurements,
    *,
    prior_id: str | None = None,
) -> SerialComparison:
    """Diff current vs prior analysis (TZ §8.4)."""
    result = SerialComparison(prior_id=prior_id)
    cur, pri = current_findings.codes(), prior_findings.codes()

    for code in sorted(cur - pri):
        if code in _SIGNIFICANT_NEW:
            ru, uz, crit = _SIGNIFICANT_NEW[code]
            result.deltas.append(SerialDelta(code, "new", ru, uz, critical=crit))
    for code in sorted(pri - cur):
        if code in _SIGNIFICANT_NEW:
            result.deltas.append(SerialDelta(
                code, "resolved",
                f"Разрешилось по сравнению с прошлой ЭКГ: {code}",
                f"Oldingi EKGga nisbatan bartaraf bo'ldi: {code}", critical=False))

    # QTc dynamics.
    cq, pq = current_m.intervals.qtc_bazett_ms, prior_m.intervals.qtc_bazett_ms
    if cq is not None and pq is not None:
        result.qtc_delta_ms = round(cq - pq, 1)
        if result.qtc_delta_ms >= _QTC_SIGNIFICANT_MS:
            result.deltas.append(SerialDelta(
                "qtc_prolongation", "changed",
                f"Удлинение QTc на {result.qtc_delta_ms} мс относительно прошлой ЭКГ",
                f"Oldingi EKGga nisbatan QTc {result.qtc_delta_ms} ms uzaydi",
                critical=cq > 500))

    if current_m.heart_rate_bpm is not None and prior_m.heart_rate_bpm is not None:
        result.hr_delta_bpm = round(current_m.heart_rate_bpm - prior_m.heart_rate_bpm, 1)

    return result
