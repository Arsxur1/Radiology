"""Cross-cutting safety layer — critical "must-not-miss" findings (TZ §10).

These detectors raise a prominent alert INDEPENDENTLY of the LLM and of the
operating mode, and always appear in the report. Priority is sensitivity
(recall) over specificity — a false alarm beats a miss (TZ §10). In autonomous
mode B, any critical finding forces escalation to a physician (TZ §2, §10).
Every alert is logged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import FindingSet

# Codes considered critical when present (mapped from Layer-4 findings).
# TZ §10 must-not-miss list; recall is prioritised over specificity.
CRITICAL_CODES = {
    "peaked_t_waves": ("Паттерн гиперкалиемии (остроконечные T)",
                       "Giperkaliyemiya patterni (uchli T)"),
    "complete_av_block": ("Полная АВ-блокада", "To'liq AV blokada"),
    "ventricular_tachycardia": ("Желудочковая тахикардия (пробежка ЖЭ)",
                                "Ventrikulyar taxikardiya (VES yugurishi)"),
    "wide_complex_tachycardia": ("Тахикардия с широкими комплексами (исключить ЖТ)",
                                 "Keng kompleksli taxikardiya (JT ni istisno qilish)"),
}


@dataclass
class CriticalAlert:
    code: str
    label_ru: str
    label_uz: str
    reason: str
    log_id: str = ""


@dataclass
class SafetyResult:
    alerts: list[CriticalAlert] = field(default_factory=list)
    audit_log: list[str] = field(default_factory=list)

    @property
    def has_critical(self) -> bool:
        return bool(self.alerts)


# 4th Universal Definition of MI, ST elevation at the J point in two contiguous
# leads (D48): 100 µV, except V2-V3 — 200 µV men >= 40 y, 250 µV men < 40 y,
# 150 µV women. Guideline constants, not tuned.
_STEMI_PAIRS = (("I", "aVL"), ("II", "III"), ("III", "aVF"), ("II", "aVF"), ("V1", "V2"),
                ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6"))


def stemi_j_point_criterion(st_median_beat: dict, sex: str | None,
                            age_years: float | None) -> bool | None:
    """True if the J-point ST on the median beat meets the UDMI STEMI criterion;
    None when no median-beat ST is available."""
    if not st_median_beat:
        return None
    elevated = set()
    for lead, (j_uv, _j60) in st_median_beat.items():
        if lead in ("V2", "V3"):
            if sex == "female":
                thr = 150.0
            else:
                thr = 250.0 if (age_years if age_years is not None else 50.0) < 40 else 200.0
        else:
            thr = 100.0
        if j_uv is not None and j_uv >= thr:
            elevated.add(lead)
    return any(a in elevated and b in elevated for a, b in _STEMI_PAIRS)


def scan_critical(findings: FindingSet, m: Measurements, *,
                  long_qt_critical_ms: float = 500.0, patient=None) -> SafetyResult:
    """Detect critical findings from findings + raw measurements (TZ §10).

    ST elevation (D48) and long QT (D44) are tiered. Long QT: the finding is reported above the sex-specific limit
    (450/460 ms), but it is a critical alert only from QTc >= long_qt_critical_ms,
    the guideline threshold for high torsades risk. Above-limit QTc marks ~20% of
    resting ECGs — GE 12SL flags the same share — so an alert at the limit carried
    no signal.
    """
    result = SafetyResult()

    def add(code: str, ru: str, uz: str, reason: str) -> None:
        alert = CriticalAlert(code=code, label_ru=ru, label_uz=uz, reason=reason,
                              log_id=f"crit-{len(result.alerts) + 1}")
        result.alerts.append(alert)
        result.audit_log.append(f"[{alert.log_id}] CRITICAL {code}: {reason}")
        f = findings.by_code(code)
        if f is not None:
            f.critical = True

    # Finding-derived critical codes.
    for code, (ru, uz) in CRITICAL_CODES.items():
        if code in findings.codes():
            add(code, ru, uz, reason=f"finding '{code}' present")

    # ST elevation is tiered (D48): the finding uses the J+60 rule; the critical
    # STEMI alert needs the guideline J-point criterion on the median beat, which
    # reproduces GE 12SL (held-out sens 0.85, false 1.9%) and fires on ~6.5% of
    # ECGs instead of ~16%. Without a median beat the alert stays (conservative).
    if "st_elevation" in findings.codes():
        sex = getattr(patient, "sex", None)
        age = getattr(patient, "age_years", None)
        stemi = stemi_j_point_criterion(m.morphology.st_median_beat, sex, age)
        if stemi is None or stemi:
            add("st_elevation", "STEMI (подозрение)", "STEMI (shubha)",
                reason="J-point ST elevation in contiguous leads (UDMI)" if stemi
                else "ST elevation; median-beat ST unavailable")

    qtc = m.intervals.qtc_bazett_ms
    if "long_qt" in findings.codes() and qtc is not None and qtc >= long_qt_critical_ms:
        add("long_qt", "Выраженное удлинение QT — риск TdP", "Keskin QT uzayishi — TdP xavfi",
            reason=f"QTc {qtc} ms >= {long_qt_critical_ms:.0f} ms")

    # Measurement-derived critical patterns (may exist without a Layer-4 finding).
    hr = m.heart_rate_bpm
    if hr is not None and hr < 40:
        add("severe_bradycardia", "Тяжёлая брадикардия", "Og'ir bradikardiya",
            reason=f"HR {hr} bpm < 40")
    # A regular narrow tachy >=140 is called SVT (Layer 4); a WIDE one at the
    # same rate is VT-until-proven-otherwise (§10). Use the same 140 threshold so
    # the narrow->SVT / wide->VT-suspect split is consistent and nothing in the
    # 140-150 band falls through to a benign sinus_tachycardia label.
    # Backstop when Layer 4 did not emit the finding (e.g. pediatric norms,
    # which have no svt_rate_min): the alert does not depend on the norm table.
    if (hr is not None and hr >= 140 and (m.intervals.qrs_ms or 0) > 120
            and "wide_complex_tachycardia" not in findings.codes()):
        add("wide_complex_tachycardia",
            "Тахикардия с широкими комплексами (исключить ЖТ)",
            "Keng kompleksli taxikardiya (JT ni istisno qilish)",
            reason=f"HR {hr} bpm with wide QRS {m.intervals.qrs_ms} ms")
    # Significant pause / asystole: longest RR gap >= 3 s (measured in Layer 3).
    if m.max_rr_ms is not None and m.max_rr_ms >= 3000:
        add("ventricular_pause",
            "Значимая пауза / асистолия",
            "Muhim pauza / asistoliya",
            reason=f"max RR gap {m.max_rr_ms:.0f} ms >= 3000 ms")

    if not result.alerts:
        result.audit_log.append("[safety] no critical findings")
    return result
