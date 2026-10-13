"""Learnable second opinion for Layer 4 (TZ §17 v2) — evidence, never authority.

The trained model (``cardiolens.ml``) gives a probability per finding code. It is
applied *after* the rule-based findings and the safety scan never sees it:

* a rule finding the model agrees with (p >= AGREE_P) gets ``ml_p`` evidence and a
  confidence bump (+CONFIDENCE_BUMP, capped);
* a rule finding the model strongly disagrees with (p <= DISAGREE_P) keeps its code
  but records the dissent as evidence (the physician sees both);
* when no rule fired and p >= SUGGEST_P for a class the model proved precise on the
  held-out **and** the cross-dataset sets at training time
  (``model.meta["suggest_classes"]``), a non-critical ``ml_suggested_<code>`` finding
  is added.
"""

from __future__ import annotations

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet

AGREE_P = 0.6
DISAGREE_P = 0.15
SUGGEST_P = 0.85
CONFIDENCE_BUMP = 0.10
CONFIDENCE_CAP = 0.95
SUGGEST_PREFIX = "ml_suggested_"

_LABELS_RU = {
    "atrial_fibrillation": "фибрилляция предсердий", "atrial_flutter": "трепетание предсердий",
    "rbbb": "блокада правой ножки", "lbbb": "блокада левой ножки",
    "first_degree_av_block": "АВ-блокада I ст.", "lvh": "гипертрофия ЛЖ",
    "low_qrs_voltage": "низкий вольтаж", "long_qt": "удлинение QT",
    "st_depression": "депрессия ST", "st_elevation": "элевация ST",
    "left_axis_deviation": "отклонение оси влево", "right_axis_deviation": "отклонение оси вправо",
    "sinus_rhythm": "синусовый ритм", "sinus_bradycardia": "брадикардия",
    "sinus_tachycardia": "тахикардия", "wpw_preexcitation": "преэкзитация (WPW)",
    "complete_av_block": "полная АВ-блокада",
}
_LABELS_UZ = {
    "atrial_fibrillation": "bo'lmachalar fibrillyatsiyasi", "atrial_flutter": "bo'lmachalar titrashi",
    "rbbb": "o'ng oyoqcha blokadasi", "lbbb": "chap oyoqcha blokadasi",
    "first_degree_av_block": "I darajali AV blokada", "lvh": "chap qorincha gipertrofiyasi",
    "low_qrs_voltage": "past voltaj", "long_qt": "QT uzayishi",
    "st_depression": "ST depressiyasi", "st_elevation": "ST elevatsiyasi",
    "left_axis_deviation": "o'qning chapga og'ishi", "right_axis_deviation": "o'qning o'ngga og'ishi",
    "sinus_rhythm": "sinus ritmi", "sinus_bradycardia": "bradikardiya",
    "sinus_tachycardia": "taxikardiya", "wpw_preexcitation": "preeksitatsiya (WPW)",
    "complete_av_block": "to'liq AV blokada",
}


def apply_ml_evidence(fs: FindingSet, m: Measurements, suggest_classes: list[str] | None = None) -> list[str]:
    """Blend ``m.ml_probabilities`` into the finding set; returns audit lines."""
    probs = m.ml_probabilities or {}
    if not probs:
        return []
    audit: list[str] = []
    present = fs.codes()
    for f in fs.findings:
        p = probs.get(f.code)
        if p is None:
            continue
        f.evidence.append(Evidence(feature="ml_p", value=f"{p:.2f}"))
        if p >= AGREE_P:
            f.confidence = round(min(CONFIDENCE_CAP, f.confidence + CONFIDENCE_BUMP), 3)
            audit.append(f"ml agrees with {f.code} (p={p:.2f}) -> confidence {f.confidence}")
        elif p <= DISAGREE_P:
            audit.append(f"ml doubts {f.code} (p={p:.2f}); rule finding kept")
    for code in suggest_classes or []:
        p = probs.get(code)
        if p is None or p < SUGGEST_P or code in present:
            continue
        fs.findings.append(Finding(
            code=SUGGEST_PREFIX + code, group="other",
            label_ru=f"Модель предполагает: {_LABELS_RU.get(code, code)} (p={p:.2f}) — проверить",
            label_uz=f"Model taxmini: {_LABELS_UZ.get(code, code)} (p={p:.2f}) — tekshiring",
            evidence=[Evidence(feature="ml_p", value=f"{p:.2f}")], confidence=round(float(p), 3),
        ))
        audit.append(f"ml suggests {code} (p={p:.2f}), no rule fired")
    return audit
