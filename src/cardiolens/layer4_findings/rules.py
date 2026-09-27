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
    # Atrial fibrillation: irregularly-irregular rhythm with absent P waves (§7.1).
    if (m.rhythm_regular is False and (m.rr_cv or 0) > 0.15
            and m.p_wave_ratio is not None and m.p_wave_ratio < 0.3):
        fs.findings.append(Finding(
            code="atrial_fibrillation", group="rhythm",
            label_ru="Фибрилляция предсердий (нерегулярный ритм, отсутствие P)",
            label_uz="Bo'lmachalar fibrillyatsiyasi (notekis ritm, P yo'q)",
            evidence=_ev(rr_cv=m.rr_cv, p_wave_ratio=m.p_wave_ratio), confidence=0.7,
        ))

    iv = m.intervals
    _, pr_hi = norms.get("intervals_ms", {}).get("pr", [None, None])
    # A PR interval needs P waves: organised P (AF gate) or a refined P onset on
    # most beats; and it is meaningless in atrial fibrillation.
    p_present = (m.p_wave_ratio is None or m.p_wave_ratio >= 0.5
                 or (m.p_onset_ratio or 0.0) >= 0.5)
    in_af = "atrial_fibrillation" in fs.codes()
    if (iv.pr_ms is not None and pr_hi and iv.pr_ms > pr_hi and p_present
            and not in_af and m.av_dissociation is not True):
        fs.findings.append(Finding(
            code="first_degree_av_block", group="interval",
            label_ru=f"АВ-блокада I степени (PR {iv.pr_ms} мс)",
            label_uz=f"I darajali AV blokada (PR {iv.pr_ms} ms)",
            evidence=_ev(pr_ms=iv.pr_ms, norm_high=pr_hi), confidence=0.85,
        ))

    _conduction_findings(m, norms, fs)
    _qt_findings(iv, norms, is_pediatric, sig, fs)

    # Complete (third-degree) AV block: atrioventricular dissociation (§7.1, §10).
    if m.av_dissociation is True:
        fs.findings.append(Finding(
            code="complete_av_block", group="rhythm",
            label_ru="Полная АВ-блокада (АВ-диссоциация)",
            label_uz="To'liq AV blokada (AV dissotsiatsiya)",
            evidence=_ev(av_dissociation=True, atrial_rate_bpm=m.atrial_rate_bpm,
                         ventricular_rate_bpm=hr), confidence=0.8,
        ))

    # Atrial flutter: regular atrial rate ~250-350/min (sawtooth) (§7.1).
    fl_lo, fl_hi = norms.get("rhythm", {}).get("atrial_flutter_bpm", [240, 350])
    if m.atrial_rate_bpm is not None and fl_lo <= m.atrial_rate_bpm <= fl_hi:
        fs.findings.append(Finding(
            code="atrial_flutter", group="rhythm",
            label_ru=f"Трепетание предсердий (частота предсердий {m.atrial_rate_bpm}/мин)",
            label_uz=f"Bo'lmachalar titrashi (bo'lmacha chastotasi {m.atrial_rate_bpm}/daq)",
            evidence=_ev(atrial_rate_bpm=m.atrial_rate_bpm), confidence=0.65,
        ))

    # A detected non-sinus rhythm (AF / flutter) replaces the sinus_* rate codes:
    # "sinus rhythm at 80" is wrong in AF, and the rate qualifier belongs to the
    # arrhythmia label instead (rapid/slow ventricular response).
    _replace_sinus_codes_if_non_sinus(fs, hr, hr_range)

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
    # Ischemia — ST / T (TZ §7.2). Runs after conduction + LVH so secondary
    # (discordant) ST-T changes can be told from primary ischemia.
    # ------------------------------------------------------------------ #
    _st_findings(m, norms, fs, sig.patient)
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


_SINUS_CODES = {"sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia"}
_NON_SINUS_CODES = {"atrial_fibrillation", "atrial_flutter", "complete_av_block"}


def _replace_sinus_codes_if_non_sinus(fs: FindingSet, hr, hr_range) -> None:
    codes = fs.codes()
    if not (codes & _NON_SINUS_CODES):
        return
    fs.findings = [f for f in fs.findings if f.code not in _SINUS_CODES]
    if hr is None or hr_range is None:
        return
    lo, hi = hr_range
    qual_ru = " с быстрым желудочковым ответом" if hr > hi else (
        " с медленным желудочковым ответом" if hr < lo else "")
    qual_uz = " tez qorincha javobi bilan" if hr > hi else (
        " sekin qorincha javobi bilan" if hr < lo else "")
    for f in fs.findings:
        if f.code in _NON_SINUS_CODES and qual_ru:
            f.label_ru += f"{qual_ru} (ЧСС {hr}/мин)"
            f.label_uz += f"{qual_uz} (YUT {hr}/daq)"


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
        # Elevation without two contiguous leads of one territory is not a STEMI
        # pattern (early repolarisation, baseline, single-lead artefact): reported
        # for the physician, never critical.
        fs.findings.append(Finding(
            code="st_elevation_nonspecific", group="ischemia",
            label_ru=f"Элевация ST без территориальной картины ({', '.join(elevated)})",
            label_uz=f"Hududiy manzarasiz ST elevatsiyasi ({', '.join(elevated)})",
            evidence=_ev(leads=",".join(elevated), threshold_uv=elev), confidence=0.5,
        ))


def _atrial_enlargement_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Left/right atrial enlargement from P-wave duration/amplitude (TZ §7.4)."""
    if fs.codes() & _NON_SINUS_CODES:
        return  # no true P waves to measure in AF / flutter
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
    # BBB typing from the terminal QRS forces in V1 (R' -> RBBB, QS/rS -> LBBB).
    # V6 is only a fallback when V1 is missing: on real LBBB the discordant ST
    # segment leaks into the terminal window and flips the V6 sign.
    v1, v6 = m.qrs_v1_terminal_positive, m.qrs_v6_terminal_positive
    term = v1 if v1 is not None else (None if v6 is None else (not v6))
    if term is True:
        fs.findings.append(Finding(
            code="rbbb", group="interval",
            label_ru=f"Полная блокада правой ножки (QRS {iv.qrs_ms} мс, RSR' в V1)",
            label_uz=f"O'ng oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms, V1 da RSR')",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="positive", v6_terminal=str(v6)),
            confidence=0.72,
        ))
    elif term is False:
        fs.findings.append(Finding(
            code="lbbb", group="interval",
            label_ru=f"Полная блокада левой ножки (QRS {iv.qrs_ms} мс)",
            label_uz=f"Chap oyoqcha to'liq blokadasi (QRS {iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="negative", v6_terminal="positive"),
            confidence=0.72,
        ))
    else:
        fs.findings.append(Finding(
            code="wide_qrs_nonspecific", group="interval",
            label_ru=f"Расширение QRS ({iv.qrs_ms} мс)",
            label_uz=f"QRS kengayishi ({iv.qrs_ms} ms)",
            evidence=_ev(qrs_ms=iv.qrs_ms, norm_high=qrs_hi), confidence=0.8,
        ))


_SECONDARY_ST_CONTEXT = {"rbbb", "lbbb", "wpw_preexcitation", "lvh"}


def _st_thresholds(norms, patient, lead: str) -> tuple[float | None, float | None]:
    """(elevation, depression) thresholds for one lead from the norm table."""
    thr = norms.get("st_uv", {})
    elev, depr = thr.get("elevation_significant"), thr.get("depression_significant")
    v23 = thr.get("elevation_v2_v3")
    if lead in ("V2", "V3") and isinstance(v23, dict) and elev:
        sex = (getattr(patient, "sex", None) or "male").lower()
        age = getattr(patient, "age_years", None)
        if sex == "female":
            elev = v23.get("female", elev)
        elif age is not None and age < 40:
            elev = v23.get("male_under_40", v23.get("male", elev))
        else:
            elev = v23.get("male", elev)
    return elev, depr


def _st_findings(m: Measurements, norms, fs: FindingSet, patient=None) -> None:
    thr = norms.get("st_uv", {})
    if not (thr.get("elevation_significant") and thr.get("depression_significant")):
        return
    plateau = float(thr.get("elevation_plateau_fraction", 0.8))
    upsloping = float(thr.get("depression_upsloping_uv", 50))
    # With a bundle branch block, pre-excitation or LVH the expected repolarisation
    # is discordant to the QRS; only concordant deviations count as primary.
    secondary_ctx = bool(fs.codes() & _SECONDARY_ST_CONTEXT)
    elevated, depressed, discordant = [], [], []
    for st in m.morphology.st:
        elev, depr = _st_thresholds(norms, patient, st.lead)
        j, j60, j80 = st.j_point_uv, st.j60_uv, st.j80_uv
        dev = 0
        # STEMI is defined by J-POINT elevation that persists into the ST segment.
        if j is not None and j >= elev and (j60 is None or j60 >= plateau * elev):
            dev = 1
        # Depression is judged in the ST segment (J+60 / J+80), horizontal or
        # downsloping; an upsloping segment (J+80 rising over J+60) is not called.
        elif j60 is not None:
            low = min(j60, j80) if j80 is not None else j60
            if low <= -depr and not (j80 is not None and j80 - j60 >= upsloping):
                dev = -1
        elif j is not None and j <= -depr:
            dev = -1
        if dev == 0:
            continue
        if (secondary_ctx and st.qrs_net_uv is not None
                and (st.qrs_net_uv > 0) != (dev > 0)):
            discordant.append(st.lead)
            # Discordant ELEVATION is the expected repolarisation of a BBB/LVH and
            # must not raise a STEMI alert. Discordant DEPRESSION is still ST
            # depression (strain pattern) — reported, and explained by the
            # LVH-strain / BBB causal chains rather than as primary ischemia.
            if dev > 0:
                continue
        (elevated if dev > 0 else depressed).append(st.lead)

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
            _territorial_st_elevation(elevated, depressed, thr.get("elevation_significant"), fs)
    if len(depressed) >= 2:
        disc = sorted(set(depressed) & set(discordant))
        fs.findings.append(Finding(
            code="st_depression", group="ischemia",
            label_ru=f"Депрессия ST ({', '.join(depressed)})"
                     + (" — дискордантная (вторичная?)" if disc else ""),
            label_uz=f"ST depressiyasi ({', '.join(depressed)})"
                     + (" — diskordant (ikkilamchi?)" if disc else ""),
            evidence=_ev(leads=",".join(depressed), threshold_uv=thr.get("depression_significant"),
                         discordant=",".join(disc) or "none"),
            confidence=0.55 if disc else 0.75,
        ))
    if len(discordant) >= 2:
        fs.findings.append(Finding(
            code="secondary_st_t_change", group="ischemia",
            label_ru=f"Вторичные (дискордантные) ST-T изменения ({', '.join(discordant)})",
            label_uz=f"Ikkilamchi (diskordant) ST-T o'zgarishlar ({', '.join(discordant)})",
            evidence=_ev(leads=",".join(discordant),
                         context=",".join(sorted(fs.codes() & _SECONDARY_ST_CONTEXT))),
            confidence=0.6,
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
