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

# Minimum RR coefficient of variation for an "irregularly-irregular" rhythm to be
# called atrial fibrillation. Tuned on a labelled PTB-XL subset: 0.15 recalls
# ~0.82 of AF at ~0.82 specificity — the knee of the sens/spec curve. AF is
# flagged at moderate confidence ("exclude AF"), so a false positive costs a
# review, not a misdiagnosis.
_AFIB_RR_CV_MIN = 0.15

# Coronary territories and their reciprocal leads (TZ §7.2, §8.1).
_TERRITORIES = [
    ("нижняя", "pastki", {"II", "III", "aVF"}, {"I", "aVL"}),
    ("передне-перегородочная", "old-septal", {"V1", "V2", "V3", "V4"}, {"II", "III", "aVF"}),
    ("боковая", "yon", {"I", "aVL", "V5", "V6"}, {"III", "aVF"}),
]


def _territory_of(elevated: set[str]) -> tuple[str, str, set[str]] | None:
    """Return (label_ru, label_uz, reciprocal_leads) for the best-matching territory."""
    best = None
    best_overlap = 0
    for ru, uz, leads, recip in _TERRITORIES:
        overlap = len(elevated & leads)
        if overlap >= 2 and overlap > best_overlap:
            best, best_overlap = (ru, uz, recip), overlap
    return best


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
    _, pr_hi = norms.get("intervals_ms", {}).get("pr", [None, None])
    if iv.pr_ms is not None and pr_hi and iv.pr_ms > pr_hi:
        fs.findings.append(Finding(
            code="first_degree_av_block", group="interval",
            label_ru=f"АВ-блокада I степени (PR {iv.pr_ms} мс)",
            label_uz=f"I darajali AV blokada (PR {iv.pr_ms} ms)",
            evidence=_ev(pr_ms=iv.pr_ms, norm_high=pr_hi), confidence=0.85,
        ))

    _conduction_findings(m, norms, fs)
    _qt_findings(iv, norms, is_pediatric, sig, fs)

    # Atrial fibrillation: irregularly-irregular rhythm (§7.1). Absent/chaotic P
    # waves are the classic confirmatory sign, but automated P-wave delineation is
    # unreliable in AF — a delineator labels a "P" for almost every beat even when
    # none is organised, so on real records ``p_wave_ratio`` stays near 1.0 and a
    # hard P-absence gate silently suppresses every detection. So the irregularly-
    # irregular RR pattern is the primary criterion; a genuinely low P-wave ratio
    # only raises confidence rather than being required.
    if m.rhythm_regular is False and (m.rr_cv or 0) >= _AFIB_RR_CV_MIN:
        p_absent = m.p_wave_ratio is not None and m.p_wave_ratio < 0.3
        fs.findings.append(Finding(
            code="atrial_fibrillation", group="rhythm",
            label_ru="Фибрилляция предсердий (нерегулярно-нерегулярный ритм)",
            label_uz="Bo'lmachalar fibrillyatsiyasi (notekis-notekis ritm)",
            evidence=_ev(rr_cv=m.rr_cv, p_wave_ratio=m.p_wave_ratio),
            confidence=0.8 if p_absent else 0.6,
        ))

    # Complete (third-degree) AV block: atrioventricular dissociation (§7.1, §10).
    # NOTE: m.av_dissociation is not yet measured by Layer 3 (always None), so this
    # rule is currently inert — see Measurements.av_dissociation / DECISIONS.md D15a.
    if m.av_dissociation is True:
        fs.findings.append(Finding(
            code="complete_av_block", group="rhythm",
            label_ru="Полная АВ-блокада (АВ-диссоциация)",
            label_uz="To'liq AV blokada (AV dissotsiatsiya)",
            evidence=_ev(av_dissociation=True, atrial_rate_bpm=m.atrial_rate_bpm,
                         ventricular_rate_bpm=hr), confidence=0.8,
        ))

    # Atrial flutter: regular atrial rate ~250-350/min (sawtooth) (§7.1).
    # NOTE: m.atrial_rate_bpm is not yet measured by Layer 3 (always None), so this
    # rule is currently inert — see Measurements.atrial_rate_bpm / DECISIONS.md D15a.
    if m.atrial_rate_bpm is not None and 240 <= m.atrial_rate_bpm <= 360:
        fs.findings.append(Finding(
            code="atrial_flutter", group="rhythm",
            label_ru=f"Трепетание предсердий (частота предсердий {m.atrial_rate_bpm}/мин)",
            label_uz=f"Bo'lmachalar titrashi (bo'lmacha chastotasi {m.atrial_rate_bpm}/daq)",
            evidence=_ev(atrial_rate_bpm=m.atrial_rate_bpm), confidence=0.65,
        ))

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
    # Axis deviation (TZ §7.4) and low QRS voltage (TZ §7.5)
    # ------------------------------------------------------------------ #
    _axis_findings(m, norms, fs)
    _low_voltage_findings(m, norms, fs)
    _atrial_enlargement_findings(m, norms, fs)

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


def _territorial_st_elevation(elevated, depressed, elev, fs: FindingSet) -> None:
    """Focal (single-territory) ST elevation → STEMI with reciprocal check (§8.1)."""
    terr = _territory_of(set(elevated))
    if terr:
        ru, uz, recip = terr
        recip_present = sorted(recip & set(depressed))
        recip_txt = f"; реципрокная депрессия в {', '.join(recip_present)}" if recip_present else ""
        fs.findings.append(Finding(
            code="st_elevation", group="ischemia",
            label_ru=f"Элевация ST, {ru} территория ({', '.join(elevated)}){recip_txt}",
            label_uz=f"ST elevatsiyasi, {uz} hudud ({', '.join(elevated)})",
            evidence=_ev(leads=",".join(elevated), territory=ru,
                         reciprocal=",".join(recip_present) or "none"), confidence=0.8,
        ))
    else:
        fs.findings.append(Finding(
            code="st_elevation", group="ischemia",
            label_ru=f"Элевация ST ({', '.join(elevated)})",
            label_uz=f"ST elevatsiyasi ({', '.join(elevated)})",
            evidence=_ev(leads=",".join(elevated), threshold_uv=elev), confidence=0.75,
        ))


def _atrial_enlargement_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Left/right atrial enlargement from P-wave duration/amplitude (TZ §7.4)."""
    pw = norms.get("p_wave", {})
    dur_hi, amp_hi = pw.get("duration_upper_ms"), pw.get("amplitude_upper_uv")
    if dur_hi and m.p_wave_duration_ms is not None and m.p_wave_duration_ms > dur_hi:
        fs.findings.append(Finding(
            code="left_atrial_enlargement", group="hypertrophy",
            label_ru=f"Увеличение левого предсердия (P-mitrale, {m.p_wave_duration_ms} мс)",
            label_uz=f"Chap bo'lmacha kattalashuvi (P-mitrale, {m.p_wave_duration_ms} ms)",
            evidence=_ev(p_wave_duration_ms=m.p_wave_duration_ms, threshold=dur_hi),
            confidence=0.65,
        ))
    if amp_hi and m.p_wave_amplitude_uv is not None and m.p_wave_amplitude_uv > amp_hi:
        fs.findings.append(Finding(
            code="right_atrial_enlargement", group="hypertrophy",
            label_ru=f"Увеличение правого предсердия (P-pulmonale, {m.p_wave_amplitude_uv} мкВ)",
            label_uz=f"O'ng bo'lmacha kattalashuvi (P-pulmonale, {m.p_wave_amplitude_uv} µV)",
            evidence=_ev(p_wave_amplitude_uv=m.p_wave_amplitude_uv, threshold=amp_hi),
            confidence=0.65,
        ))


def _axis_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """QRS-axis deviation (TZ §7.4)."""
    axis = m.axes.qrs_deg
    ad = norms.get("axis_deg", {})
    rng = ad.get("qrs_normal")
    if axis is None or not rng:
        return
    lo, hi = rng
    ext_lo, ext_hi = ad.get("extreme_low"), ad.get("extreme_high")
    if ext_lo is not None and (axis < ext_lo or (ext_hi is not None and axis > ext_hi)):
        fs.findings.append(Finding(
            code="extreme_axis", group="hypertrophy",
            label_ru=f"Резкое отклонение оси (ось QRS {axis}°)",
            label_uz=f"O'qning keskin og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis), confidence=0.7,
        ))
    elif axis < lo:
        fs.findings.append(Finding(
            code="left_axis_deviation", group="hypertrophy",
            label_ru=f"Отклонение оси влево (ось QRS {axis}°)",
            label_uz=f"O'qning chapga og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis, norm_low=lo), confidence=0.7,
        ))
    elif axis > hi:
        fs.findings.append(Finding(
            code="right_axis_deviation", group="hypertrophy",
            label_ru=f"Отклонение оси вправо (ось QRS {axis}°)",
            label_uz=f"O'qning o'ngga og'ishi (QRS o'qi {axis}°)",
            evidence=_ev(qrs_axis_deg=axis, norm_high=hi), confidence=0.7,
        ))


def _low_voltage_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Low QRS voltage in the limb leads (TZ §7.5)."""
    thr = norms.get("voltage_uv", {}).get("low_voltage_limb")
    amp = m.qrs_amplitude_limb_uv
    if thr and amp is not None and amp < thr:
        fs.findings.append(Finding(
            code="low_qrs_voltage", group="other",
            label_ru=f"Низкий вольтаж QRS (амплитуда {amp} мкВ)",
            label_uz=f"Past QRS voltaji (amplituda {amp} µV)",
            evidence=_ev(qrs_amplitude_limb_uv=amp, threshold=thr), confidence=0.7,
        ))


def _conduction_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Classify a wide QRS into WPW / RBBB / LBBB / nonspecific (TZ §7.1)."""
    iv = m.intervals
    qrs_hi = norms.get("intervals_ms", {}).get("qrs", [None, None])[1]
    pr_lo = norms.get("intervals_ms", {}).get("pr", [None, None])[0]
    if iv.qrs_ms is None or not qrs_hi or iv.qrs_ms <= qrs_hi:
        return
    # Short PR + wide QRS -> pre-excitation (WPW): the wide QRS is explained by a
    # delta wave, not by a bundle branch block (TZ §8.1).
    if pr_lo and iv.pr_ms is not None and iv.pr_ms < pr_lo:
        fs.findings.append(Finding(
            code="wpw_preexcitation", group="interval",
            label_ru=f"Преэкзитация (WPW): короткий PR {iv.pr_ms} мс + широкий QRS {iv.qrs_ms} мс",
            label_uz=f"Preeksitatsiya (WPW): qisqa PR {iv.pr_ms} ms + keng QRS {iv.qrs_ms} ms",
            evidence=_ev(pr_ms=iv.pr_ms, qrs_ms=iv.qrs_ms), confidence=0.7,
        ))
        return
    # A *complete* bundle-branch block needs a fully widened QRS. Between the
    # upper norm and the complete-block threshold the QRS is only a nonspecific
    # (or incomplete) conduction delay — typing it as a full RBBB/LBBB is the main
    # source of false positives on real records.
    block_min = norms.get("intervals_ms", {}).get("qrs_complete_block")
    term = m.qrs_v1_terminal_positive
    complete = block_min is not None and iv.qrs_ms >= block_min
    if complete and term is True:
        fs.findings.append(Finding(
            code="rbbb", group="interval",
            label_ru=f"Полная блокада правой ножки (QRS {iv.qrs_ms} мс, RSR' в V1)",
            label_uz=f"O'ng oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms, V1 da RSR')",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="positive"), confidence=0.72,
        ))
    elif complete and term is False:
        fs.findings.append(Finding(
            code="lbbb", group="interval",
            label_ru=f"Полная блокада левой ножки (QRS {iv.qrs_ms} мс)",
            label_uz=f"Chap oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="negative"), confidence=0.72,
        ))
    else:
        fs.findings.append(Finding(
            code="wide_qrs_nonspecific", group="interval",
            label_ru=f"Неспецифическое расширение QRS ({iv.qrs_ms} мс)",
            label_uz=f"Nospetsifik QRS kengayishi ({iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, norm_high=qrs_hi), confidence=0.8,
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
        # Diffuse ST elevation across >=2 coronary territories (and no single
        # reciprocal territory) points to pericarditis, not a focal STEMI — a
        # classic mimic distinction (ecglibrary / 150 ECG Problems).
        n_terr = sum(1 for _, _, leads, _ in _TERRITORIES if len(set(elevated) & leads) >= 2)
        if n_terr >= 2 and len(depressed) == 0:
            fs.findings.append(Finding(
                code="pericarditis", group="other",
                label_ru=f"Диффузная элевация ST — перикардит? ({', '.join(elevated)})",
                label_uz=f"Diffuz ST elevatsiyasi — perikardit? ({', '.join(elevated)})",
                evidence=_ev(leads=",".join(elevated), territories=n_terr,
                             reciprocal="none"), confidence=0.6,
            ))
        else:
            _territorial_st_elevation(elevated, depressed, elev, fs)
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
