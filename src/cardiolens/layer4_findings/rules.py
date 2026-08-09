"""Layer 4 — findings rule engine (TZ §7).

Turns Layer-3 measurements into evidence-backed, bilingual findings across the
MVP subset of groups: rhythm, intervals/QT, ischemia (ST/T), hypertrophy (LVH),
plus morphology/artifact codes that feed the Layer-5 causal engine.

Thresholds come from the age/sex norm tables (TZ §6) — not hard-coded. Findings
this layer cannot support are simply not emitted (TZ §0, no fabrication).
"""

from __future__ import annotations

from typing import Any

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.contracts.signal import SignalRepresentation

# Precordial leads where juvenile T-inversion is a normal variant in children.
_PEDIATRIC_TINV_OK = {"V1", "V2", "V3", "V4"}


def _ev(**kv: Any) -> list[Evidence]:
    return [Evidence(feature=k, value=str(v)) for k, v in kv.items()]


def _hr_range(norms: dict[str, Any], patient) -> tuple[float, float] | None:
    """Resolve HR norm range, honouring pediatric age-banding (TZ §6)."""
    if "heart_rate_bpm" in norms:
        lo, hi = norms["heart_rate_bpm"]
        return float(lo), float(hi)
    bands = norms.get("heart_rate_bpm_by_age", [])
    age_days = patient.age_days
    age_years = patient.age_years
    for band in bands:
        if "age_days_max" in band and age_days is not None and age_days <= band["age_days_max"]:
            return tuple(band["range"])
        if "age_years_max" in band and age_years is not None and age_years <= band["age_years_max"]:
            return tuple(band["range"])
    if bands:
        return tuple(bands[-1]["range"])
    return None


def detect_findings(
    sig: SignalRepresentation,
    m: Measurements,
    *,
    norms: dict[str, Any],
    la_ra_reversed: bool = False,
) -> FindingSet:
    fs = FindingSet()
    is_pediatric = norms.get("age_group") == "pediatric"

    # ------------------------------------------------------------------ #
    # Rhythm (TZ §7.1)
    # ------------------------------------------------------------------ #
    hr = m.heart_rate_bpm
    hr_range = _hr_range(norms, sig.patient)
    if hr is not None and hr_range is not None:
        lo, hi = hr_range
        if hr < lo:
            fs.findings.append(Finding(
                code="sinus_bradycardia", group="rhythm",
                label_ru=f"Брадикардия (ЧСС {hr}/мин)",
                label_uz=f"Bradikardiya (YUT {hr}/daq)",
                evidence=_ev(heart_rate_bpm=hr, norm_low=lo), confidence=0.9,
            ))
        elif hr > hi:
            fs.findings.append(Finding(
                code="sinus_tachycardia", group="rhythm",
                label_ru=f"Тахикардия (ЧСС {hr}/мин)",
                label_uz=f"Taxikardiya (YUT {hr}/daq)",
                evidence=_ev(heart_rate_bpm=hr, norm_high=hi), confidence=0.9,
            ))
        else:
            fs.findings.append(Finding(
                code="sinus_rhythm", group="rhythm",
                label_ru=f"Синусовый ритм (ЧСС {hr}/мин)",
                label_uz=f"Sinus ritmi (YUT {hr}/daq)",
                evidence=_ev(heart_rate_bpm=hr), confidence=0.85,
            ))

    if m.rhythm_regular is False:
        fs.findings.append(Finding(
            code="irregular_rhythm", group="rhythm",
            label_ru="Нерегулярный ритм (исключить ФП)",
            label_uz="Notekis ritm (AF ni istisno qilish)",
            evidence=_ev(rr_std_ms=m.rr_std_ms), confidence=0.6,
        ))

    # ------------------------------------------------------------------ #
    # Intervals / QT (TZ §7.3)
    # ------------------------------------------------------------------ #
    iv = m.intervals
    pr_lo, pr_hi = norms.get("intervals_ms", {}).get("pr", [None, None])
    if iv.pr_ms is not None and pr_hi and iv.pr_ms > pr_hi:
        fs.findings.append(Finding(
            code="first_degree_av_block", group="interval",
            label_ru=f"АВ-блокада I степени (PR {iv.pr_ms} мс)",
            label_uz=f"I darajali AV blokada (PR {iv.pr_ms} ms)",
            evidence=_ev(pr_ms=iv.pr_ms, norm_high=pr_hi), confidence=0.85,
        ))

    qrs_lo, qrs_hi = norms.get("intervals_ms", {}).get("qrs", [None, None])
    if iv.qrs_ms is not None and qrs_hi and iv.qrs_ms > qrs_hi:
        fs.findings.append(Finding(
            code="wide_qrs_nonspecific", group="interval",
            label_ru=f"Расширение QRS ({iv.qrs_ms} мс)",
            label_uz=f"QRS kengayishi ({iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, norm_high=qrs_hi), confidence=0.8,
        ))

    _qt_findings(iv, norms, is_pediatric, sig, fs)

    # ------------------------------------------------------------------ #
    # Ischemia — ST / T (TZ §7.2)
    # ------------------------------------------------------------------ #
    _st_findings(m, norms, fs)
    _t_inversion_findings(m, is_pediatric, fs)

    # Peaked T-waves feed the hyperkalemia causal chain.
    if len(m.morphology.peaked_t_leads) >= 2:
        fs.findings.append(Finding(
            code="peaked_t_waves", group="other",
            label_ru="Высокие остроконечные зубцы T",
            label_uz="Baland uchli T to'lqinlar",
            evidence=_ev(leads=",".join(m.morphology.peaked_t_leads)), confidence=0.7,
        ))
    if len(m.morphology.flat_t_leads) >= 3:
        fs.findings.append(Finding(
            code="flat_t_waves", group="other",
            label_ru="Уплощение зубцов T",
            label_uz="T to'lqinlarining yassilanishi",
            evidence=_ev(leads=",".join(m.morphology.flat_t_leads)), confidence=0.6,
        ))
    if m.morphology.u_wave_leads:
        fs.findings.append(Finding(
            code="u_waves", group="other",
            label_ru="U-волны",
            label_uz="U to'lqinlar",
            evidence=_ev(leads=",".join(m.morphology.u_wave_leads)), confidence=0.6,
        ))

    # ------------------------------------------------------------------ #
    # Hypertrophy — LVH (TZ §7.4)
    # ------------------------------------------------------------------ #
    lvh_thr = norms.get("voltage_uv", {}).get("sokolow_lyon_lvh")
    if m.sokolow_lyon_uv is not None and lvh_thr and m.sokolow_lyon_uv > lvh_thr:
        fs.findings.append(Finding(
            code="lvh", group="hypertrophy",
            label_ru=f"Гипертрофия ЛЖ (Соколов-Лайон {m.sokolow_lyon_uv} мкВ)",
            label_uz=f"Chap qorincha gipertrofiyasi (Sokolow-Lyon {m.sokolow_lyon_uv} µV)",
            evidence=_ev(sokolow_lyon_uv=m.sokolow_lyon_uv, threshold=lvh_thr),
            confidence=0.8,
        ))

    # ------------------------------------------------------------------ #
    # Artifact hypothesis from Layer 2 (TZ §5, feeds §8.3)
    # ------------------------------------------------------------------ #
    if la_ra_reversed:
        fs.findings.append(Finding(
            code="lead_reversal_la_ra", group="other",
            label_ru="Подозрение на реверс электродов LA/RA",
            label_uz="LA/RA elektrodlar almashinuvi shubhasi",
            evidence=_ev(lead_I="inverted", aVR="upright"), confidence=0.65,
        ))

    return fs


def _qt_findings(iv, norms, is_pediatric, sig, fs: FindingSet) -> None:
    qtc = iv.qtc_bazett_ms
    if qtc is None:
        return
    qtc_norms = norms.get("qtc_ms", {})
    if is_pediatric:
        upper = qtc_norms.get("upper")
        short_lower = qtc_norms.get("short_lower")
    else:
        sex = (sig.patient.sex or "male").lower()
        upper = qtc_norms.get("female_upper" if sex == "female" else "male_upper")
        short_lower = qtc_norms.get("short_lower")
    if upper and qtc > upper:
        fs.findings.append(Finding(
            code="long_qt", group="interval",
            label_ru=f"Удлинение QT (QTc {qtc} мс)",
            label_uz=f"QT uzayishi (QTc {qtc} ms)",
            evidence=_ev(qtc_bazett_ms=qtc, upper=upper), confidence=0.85,
        ))
    elif short_lower and qtc < short_lower:
        fs.findings.append(Finding(
            code="short_qt", group="interval",
            label_ru=f"Укорочение QT (QTc {qtc} мс)",
            label_uz=f"QT qisqarishi (QTc {qtc} ms)",
            evidence=_ev(qtc_bazett_ms=qtc, lower=short_lower), confidence=0.7,
        ))


def _st_findings(m: Measurements, norms, fs: FindingSet) -> None:
    thr = norms.get("st_uv", {})
    elev = thr.get("elevation_significant")
    depr = thr.get("depression_significant")
    if not (elev and depr):
        return
    # STEMI is clinically defined by J-POINT elevation. Keying on the J point
    # (not J+60) avoids false positives from T-wave encroachment at high rates.
    elevated, depressed = [], []
    for st in m.morphology.st:
        if st.j_point_uv is None:
            continue
        if st.j_point_uv >= elev:
            elevated.append(st.lead)
        elif st.j_point_uv <= -depr:
            depressed.append(st.lead)
    if len(elevated) >= 2:
        fs.findings.append(Finding(
            code="st_elevation", group="ischemia",
            label_ru=f"Элевация ST ({', '.join(elevated)})",
            label_uz=f"ST elevatsiyasi ({', '.join(elevated)})",
            evidence=_ev(leads=",".join(elevated), threshold_uv=elev), confidence=0.75,
        ))
    if len(depressed) >= 2:
        fs.findings.append(Finding(
            code="st_depression", group="ischemia",
            label_ru=f"Депрессия ST ({', '.join(depressed)})",
            label_uz=f"ST depressiyasi ({', '.join(depressed)})",
            evidence=_ev(leads=",".join(depressed), threshold_uv=depr), confidence=0.75,
        ))


def _t_inversion_findings(m: Measurements, is_pediatric: bool, fs: FindingSet) -> None:
    inv = [l for l in m.morphology.inverted_t_leads if l != "aVR"]
    if not inv:
        return
    # Pediatric caveat (TZ §7.2): juvenile T-inversion in V1-V4 is NORMAL.
    if is_pediatric:
        inv = [l for l in inv if l not in _PEDIATRIC_TINV_OK]
        if not inv:
            fs.findings.append(Finding(
                code="juvenile_t_pattern", group="other",
                label_ru="Ювенильная инверсия T (V1–V4) — норма у детей",
                label_uz="Yuvenil T inversiyasi (V1–V4) — bolalarda norma",
                evidence=_ev(note="pediatric_normal_variant"), confidence=0.7,
            ))
            return
    if len(inv) >= 2:
        fs.findings.append(Finding(
            code="t_inversion", group="ischemia",
            label_ru=f"Инверсия зубца T ({', '.join(inv)})",
            label_uz=f"T to'lqin inversiyasi ({', '.join(inv)})",
            evidence=_ev(leads=",".join(inv)), confidence=0.65,
        ))
