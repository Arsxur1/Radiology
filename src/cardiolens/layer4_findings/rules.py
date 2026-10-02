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
# D15z: the std-based CV is inflated by a few ectopic beats (sinus + PVC/PAC was
# the largest AFib false-positive pool on PTB-XL fold 10, D15x), so the call also
# requires broad robust dispersion (rr_mad_cv). With that gate the CV threshold
# can drop to 0.12. Chosen on development data only (Chapman/Ningbo subset +
# PTB-XL outside test fold 10): TP 129 -> 142, FP 182 -> 157 versus CV >= 0.15.
_AFIB_RR_CV_MIN = 0.12
_AFIB_RR_MAD_CV_MIN = 0.06
# Too few beats for a robust dispersion (rr_mad_cv undefined): previous rule.
_AFIB_RR_CV_MIN_NO_MAD = 0.15
# Incomplete RBBB: minimum measured QRS (D27).
_IRBBB_QRS_MIN_MS = 80.0

# Coronary territories and their reciprocal leads (TZ §7.2, §8.1).
_TERRITORIES = [
    ("нижняя", "pastki", {"II", "III", "aVF"}, {"I", "aVL"}),
    ("передне-перегородочная", "old-septal", {"V1", "V2", "V3", "V4"}, {"II", "III", "aVF"}),
    ("боковая", "yon", {"I", "aVL", "V5", "V6"}, {"III", "aVF"}),
]

# Wide QRS with a terminal R in V1 at or above this width is read as RBBB, not WPW (D58).
_WPW_RBBB_QRS_MS = 130.0
# Delta-wave rule (D59): global PR below this and initial/peak QRS energy below that.
_WPW_PR_MAX_MS = 130.0
_DELTA_RATIO_MAX = 0.22


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
            # A regular narrow-complex tachycardia at/above svt_rate_min is SVT
            # (or atrial tachycardia / flutter with 2:1), NOT sinus tachycardia —
            # resting sinus tachy rarely exceeds ~140/min (§7.1). The narrow gate
            # keeps a wide-complex tachy in VT territory (§10, handled by the beat
            # classifier); the regular gate keeps AFib-RVR out (it fires
            # atrial_fibrillation instead). Adult-only (svt_rate_min absent from
            # the pediatric norms, where high rates are physiologic).
            svt_min = norms.get("svt_rate_min")
            block = norms.get("intervals_ms", {}).get("qrs_complete_block") or 120
            qrs = m.intervals.qrs_ms
            narrow = qrs is not None and qrs < block
            wide = qrs is not None and qrs >= block
            if not is_pediatric and svt_min and hr >= svt_min and wide:
                # A WIDE tachycardia at the same rate is VT until proven otherwise
                # (§10; the safety layer raises the critical alert). Not labelled
                # sinus: on INCART a 175/min VT run was reported as sinus tachycardia.
                fs.findings.append(Finding(
                    code="wide_complex_tachycardia", group="rhythm",
                    label_ru=f"Тахикардия с широкими QRS (ЧСС {hr}/мин, QRS {qrs} мс) — "
                             f"исключить желудочковую тахикардию",
                    label_uz=f"Keng QRS li taxikardiya (YUT {hr}/daq, QRS {qrs} ms) — "
                             f"ventrikulyar taxikardiyani istisno qilish",
                    evidence=_ev(heart_rate_bpm=hr, qrs_ms=qrs, rate_min=svt_min,
                                 qrs_min=block), confidence=0.7,
                ))
            elif (not is_pediatric and svt_min and hr >= svt_min
                    and m.rhythm_regular is not False and narrow):
                fs.findings.append(Finding(
                    code="supraventricular_tachycardia", group="rhythm",
                    label_ru=f"Суправентрикулярная тахикардия (ЧСС {hr}/мин, "
                             f"регулярная узкокомплексная)",
                    label_uz=f"Supraventrikulyar taxikardiya (YUT {hr}/daq)",
                    evidence=_ev(heart_rate_bpm=hr, svt_rate_min=svt_min,
                                 qrs_ms=qrs, regular="yes"), confidence=0.7,
                ))
            else:
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
    intervals_norm = norms.get("intervals_ms", {})
    _, pr_hi = intervals_norm.get("pr", [None, None])
    # First-degree AV block: prefer multi-lead agreement (a single lead under-
    # measures PR, missing ~half of true blocks). Fires when >=min_leads agree
    # PR > upper norm; falls back to the single measured PR if no per-lead data.
    min_leads = intervals_norm.get("pr_prolonged_min_leads")
    prolonged_leads = [ld for ld, v in (m.pr_ms_by_lead or {}).items()
                       if pr_hi and v > pr_hi]
    single = iv.pr_ms is not None and pr_hi and iv.pr_ms > pr_hi
    # Union: the primary lead II prolonged, OR >=min_leads leads agree. The union
    # never loses a lead-II detection and adds cases II alone under-measures.
    multi = bool(pr_hi and min_leads and len(prolonged_leads) >= min_leads)
    # The global (12-lead median beat) PR is already a multi-lead measurement and
    # far less noisy (D30): it decides alone; the per-lead union is the fallback.
    prolonged = bool(single) if m.pr_global else (single or multi)
    if prolonged:
        pr_repr = iv.pr_ms if iv.pr_ms is not None else max(
            (m.pr_ms_by_lead or {}).values(), default=None)
        leads_txt = ",".join(prolonged_leads) if prolonged_leads else "II"
        fs.findings.append(Finding(
            code="first_degree_av_block", group="interval",
            label_ru=f"АВ-блокада I степени (PR {pr_repr} мс; отведения {leads_txt})",
            label_uz=f"I darajali AV blokada (PR {pr_repr} ms; {leads_txt})",
            evidence=_ev(pr_ms=pr_repr, norm_high=pr_hi,
                         prolonged_leads=leads_txt), confidence=0.85,
        ))

    _conduction_findings(m, norms, fs, is_pediatric)
    _qt_findings(iv, norms, is_pediatric, sig, fs)
    _q_wave_findings(m, norms, fs)

    # Atrial fibrillation: irregularly-irregular rhythm (§7.1). Absent/chaotic P
    # waves are the classic confirmatory sign, but automated P-wave delineation is
    # unreliable in AF — a delineator labels a "P" for almost every beat even when
    # none is organised, so on real records ``p_wave_ratio`` stays near 1.0 and a
    # hard P-absence gate silently suppresses every detection. So the irregularly-
    # irregular RR pattern is the primary criterion; a genuinely low P-wave ratio
    # only raises confidence rather than being required.
    # A few ectopic beats on a sinus background inflate rr_cv but leave most RR
    # clustered at the median (low rr_mad_cv); AF spreads RR broadly.
    if m.rr_mad_cv is None:
        afib_rr = (m.rr_cv or 0) >= _AFIB_RR_CV_MIN_NO_MAD
    else:
        afib_rr = (m.rr_cv or 0) >= _AFIB_RR_CV_MIN and m.rr_mad_cv >= _AFIB_RR_MAD_CV_MIN
    if m.rhythm_regular is False and afib_rr:
        p_absent = m.p_wave_ratio is not None and m.p_wave_ratio < 0.3
        fs.findings.append(Finding(
            code="atrial_fibrillation", group="rhythm",
            label_ru="Фибрилляция предсердий (нерегулярно-нерегулярный ритм)",
            label_uz="Bo'lmachalar fibrillyatsiyasi (notekis-notekis ritm)",
            evidence=_ev(rr_cv=m.rr_cv, rr_mad_cv=m.rr_mad_cv, p_wave_ratio=m.p_wave_ratio),
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

    # Right ventricular hypertrophy: dominant R in V1 (R/S > threshold). A
    # dominant V1 R also comes from RBBB (excluded here, already typed) and is a
    # normal infant variant (pediatric norms omit the `rvh` key, so this is
    # adult-only). Right-axis deviation is supportive and raises confidence (§7.5).
    rvh_thr = norms.get("rvh", {}).get("v1_rs_ratio")
    if (rvh_thr and m.qrs_v1_rs_ratio is not None and m.qrs_v1_rs_ratio > rvh_thr
            and fs.by_code("rbbb") is None):
        right_axis = fs.by_code("right_axis_deviation") is not None
        fs.findings.append(Finding(
            code="rvh", group="hypertrophy",
            label_ru=f"Гипертрофия ПЖ (доминантный R в V1, R/S {m.qrs_v1_rs_ratio})",
            label_uz=f"O'ng qorincha gipertrofiyasi (V1 da dominant R, R/S {m.qrs_v1_rs_ratio})",
            evidence=_ev(v1_rs_ratio=m.qrs_v1_rs_ratio, threshold=rvh_thr,
                         right_axis="yes" if right_axis else "no"),
            confidence=0.75 if right_axis else 0.6,
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

    # In a wide-complex tachycardia the "P" before each QRS and the initial QRS
    # forces are not interpretable: a short "PR" is not pre-excitation, PR is not
    # AV conduction, and Q waves of a ventricular rhythm are not infarct Q waves.
    if fs.by_code("wide_complex_tachycardia") is not None:
        fs.findings = [f for f in fs.findings if f.code not in _WCT_EXCLUDES]

    return fs


_WCT_EXCLUDES = ("wpw_preexcitation", "first_degree_av_block", "pathological_q_waves")


def _q_wave_findings(m: Measurements, norms, fs: FindingSet) -> None:
    """Pathological Q waves in >= 2 contiguous leads — possible prior infarction (D36).

    Q criteria are not interpretable with LBBB or pre-excitation (both create QS
    complexes and pseudo-Q waves), so those suppress the finding here; pacing is
    handled by the rhythm hierarchy in the pipeline.
    """
    qn = norms.get("q_wave")
    if not qn or not m.morphology.q_waves:
        return
    if fs.by_code("lbbb") is not None or fs.by_code("wpw_preexcitation") is not None:
        return
    patho = set()
    for lead, (q_ms, q_depth, r_amp, qs) in m.morphology.q_waves.items():
        if lead in ("aVR", "V1"):
            continue
        if lead in ("V2", "V3"):
            hit = qs or q_ms > qn["v2v3_q_ms_max"]
        else:
            hit = qs or (q_ms >= qn["q_ms_min"] and q_depth >= qn["q_depth_uv_min"]
                         and q_depth >= qn["q_depth_r_ratio_min"] * max(r_amp, 1.0))
        if hit:
            patho.add(lead)
    pairs = [(a, b) for a, b in _Q_CONTIGUOUS if a in patho and b in patho]
    if not pairs:
        return
    leads = [ld for ld in _Q_LEAD_ORDER if ld in {x for p in pairs for x in p}]
    walls = [(ru, uz) for ru, uz, group in _Q_WALLS if group & set(leads)]
    wall_ru = ", ".join(ru for ru, _ in walls) + (" стенки" if len(walls) > 1 else " стенка")
    wall_uz = ", ".join(uz for _, uz in walls) + (" devorlar" if len(walls) > 1 else " devor")
    leads_txt = ",".join(leads)
    fs.findings.append(Finding(
        code="pathological_q_waves", group="ischemia",
        label_ru=f"Патологические зубцы Q ({leads_txt}) — возможен перенесённый инфаркт "
                 f"({wall_ru})",
        label_uz=f"Patologik Q to'lqinlar ({leads_txt}) — o'tkazilgan infarkt ehtimoli "
                 f"({wall_uz})",
        evidence=_ev(leads=leads_txt, **{f"q_{ld}": m.morphology.q_waves[ld] for ld in leads}),
        confidence=0.6,
    ))


# Contiguous lead pairs and walls for pathological Q waves (D36).
_Q_CONTIGUOUS = [("I", "aVL"), ("II", "III"), ("III", "aVF"), ("II", "aVF"),
                 ("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6")]
_Q_LEAD_ORDER = ["I", "II", "III", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_Q_WALLS = [("нижняя", "pastki", {"II", "III", "aVF"}),
            ("передняя", "old", {"V1", "V2", "V3", "V4"}),
            ("боковая", "yon", {"I", "aVL", "V5", "V6"})]


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
    # Plain QTc against the sex-specific limit. A QRS-adjusted limit + multi-lead QT
    # median improved dev data but lost sensitivity on held-out Georgia (0.48 ->
    # 0.38) — reverted for this critical finding (D28).
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
    """Low QRS voltage (TZ §7.5): every limb lead under the limb limit or every
    precordial lead under the precordial limit, on the median beat (D61). The former
    median-of-I/II/III measure flagged 8% of ECGs (fold-10 PPV 0.07; D61, D62); it is used only
    when the median-beat amplitudes are unavailable."""
    v = norms.get("voltage_uv", {})
    thr, thr_prec = v.get("low_voltage_limb"), v.get("low_voltage_precordial")
    limb, prec = m.qrs_amplitude_limb_max_uv, m.qrs_amplitude_precordial_max_uv
    if limb is None and prec is None:
        amp = m.qrs_amplitude_limb_uv
        if thr and amp is not None and amp < thr:
            fs.findings.append(Finding(
                code="low_qrs_voltage", group="other",
                label_ru=f"Низкий вольтаж QRS (амплитуда {amp} мкВ)",
                label_uz=f"Past QRS voltaji (amplituda {amp} µV)",
                evidence=_ev(qrs_amplitude_limb_uv=amp, threshold=thr), confidence=0.6,
            ))
        return
    low_limb = bool(thr and limb is not None and limb < thr)
    low_prec = bool(thr_prec and prec is not None and prec < thr_prec)
    if not (low_limb or low_prec):
        return
    where_ru = " и ".join(x for x, ok in (("в отведениях от конечностей", low_limb),
                                          ("в грудных отведениях", low_prec)) if ok)
    where_uz = " va ".join(x for x, ok in (("oyoq-qo'l tarmoqlarida", low_limb),
                                          ("ko'krak tarmoqlarida", low_prec)) if ok)
    fs.findings.append(Finding(
        code="low_qrs_voltage", group="other",
        label_ru=f"Низкий вольтаж QRS {where_ru} (наибольший QRS: конечности {limb} мкВ, "
                 f"грудные {prec} мкВ)",
        label_uz=f"Past QRS voltaji {where_uz} (eng katta QRS: {limb} / {prec} µV)",
        evidence=_ev(qrs_limb_max_uv=limb, qrs_precordial_max_uv=prec,
                     limb_threshold=thr, precordial_threshold=thr_prec), confidence=0.7,
    ))


def _conduction_findings(m: Measurements, norms, fs: FindingSet, is_pediatric: bool = False) -> None:
    """Classify a wide QRS into WPW / RBBB / LBBB / nonspecific (TZ §7.1)."""
    iv = m.intervals
    qrs_hi = norms.get("intervals_ms", {}).get("qrs", [None, None])[1]
    pr_lo = norms.get("intervals_ms", {}).get("pr", [None, None])[0]
    if iv.qrs_ms is None or not qrs_hi:
        return
    # Pre-excitation (WPW) from its two signs on the median beat (D59): a short global
    # PR and a slow, slurred QRS upstroke (delta wave). No QRS-width requirement — the
    # delta wave often leaves the QRS under 110 ms. Development data (PTB-XL folds 1-9,
    # with the RBBB exclusion): 27/71 WPW at 2 false calls; the PR-only rule found 16 at 12.
    ratio = m.qrs_initial_energy_ratio
    # RBBB morphology (terminal R in V1, QRS >= 130 ms) goes to the bundle-branch rule:
    # without this, D59 hid 2 RBBB on each held-out set (D60).
    rbbb_like = m.qrs_v1_terminal_positive is True and iv.qrs_ms >= _WPW_RBBB_QRS_MS
    if (ratio is not None and iv.pr_ms is not None and iv.pr_ms < _WPW_PR_MAX_MS
            and ratio < _DELTA_RATIO_MAX and not rbbb_like):
        fs.findings.append(Finding(
            code="wpw_preexcitation", group="interval",
            label_ru=f"Преэкзитация (WPW): короткий PR {iv.pr_ms} мс + дельта-волна "
                     f"(медленное начало QRS), QRS {iv.qrs_ms} мс",
            label_uz=f"Preeksitatsiya (WPW): qisqa PR {iv.pr_ms} ms + delta to'lqin, QRS {iv.qrs_ms} ms",
            evidence=_ev(pr_ms=iv.pr_ms, qrs_ms=iv.qrs_ms, qrs_initial_energy_ratio=ratio),
            confidence=0.7,
        ))
        return
    block_min = norms.get("intervals_ms", {}).get("qrs_complete_block") or qrs_hi
    # Incomplete RBBB (D27): RBBB morphology (terminal R' in V1) with a partly
    # widened QRS. Adults only — an rSR' in V1 is a common normal variant in
    # children, and a narrow (<80 ms) rSr' is a normal variant at any age. The
    # lower bound sits below the textbook 110 ms because the energy width reads
    # incomplete RBBB short (median 100 ms). Development data (PTB-XL outside
    # fold 10): sens 0.47, spec 0.968 (0.52 / 0.967 with no lower bound).
    if _IRBBB_QRS_MIN_MS <= iv.qrs_ms < block_min and m.qrs_v1_terminal_positive is True \
            and not is_pediatric \
            and not (pr_lo and iv.pr_ms is not None and iv.pr_ms < pr_lo and iv.qrs_ms > qrs_hi):
        fs.findings.append(Finding(
            code="incomplete_rbbb", group="interval",
            label_ru=f"Неполная блокада правой ножки (QRS {iv.qrs_ms} мс, rSR' в V1)",
            label_uz=f"O'ng oyoqcha to'liqsiz blokadasi (QRS {iv.qrs_ms} ms, V1 da rSR')",
            evidence=_ev(qrs_ms=iv.qrs_ms, v1_terminal="positive"), confidence=0.6,
        ))
        return
    if iv.qrs_ms <= qrs_hi:
        return
    # Short PR + wide QRS -> pre-excitation (WPW): the wide QRS is explained by a
    # delta wave, not by a bundle branch block (TZ §8.1). A terminal R in V1 with a
    # QRS >= 130 ms is RBBB morphology: the global PR (D30) is short on some RBBB
    # records and WPW would hide the block, so the bundle-branch rule decides (D58).
    rbbb_morphology = m.qrs_v1_terminal_positive is True and iv.qrs_ms >= _WPW_RBBB_QRS_MS
    if (ratio is None and pr_lo and iv.pr_ms is not None and iv.pr_ms < pr_lo
            and not rbbb_morphology):
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
    # Complete-block threshold; if a norms config omits it, fall back to the
    # upper QRS norm (prior behaviour) rather than disabling all BBB detection.
    term = m.qrs_v1_terminal_positive
    complete = iv.qrs_ms >= block_min
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
    # Both ST deviations are read at J+60 ms (SCP-ECG ST-amplitude convention).
    # The J point alone under-reads elevation (sens 0.14 vs 0.29) and, sitting in
    # the S-wave / junctional dip, over-reads depression on normals (spec 0.47 vs
    # 0.70 at J+60) — junctional (upsloping) depression is benign, ischaemic
    # depression persists to J+60/J+80 (D15j/D15k). Precordial V2-V3 take a higher
    # elevation threshold — early-repolarisation routinely lifts them 0.1-0.2 mV
    # in healthy hearts, so ACC/ESC require 0.2 mV there.
    elev_v2v3 = thr.get("elevation_significant_v2v3", elev)
    elevated, depressed = [], []
    for st in m.morphology.st:
        if st.j60_uv is None:
            continue
        lead_thr = elev_v2v3 if st.lead in ("V2", "V3") else elev
        if st.j60_uv >= lead_thr:
            elevated.append(st.lead)
        elif st.j60_uv <= -depr:
            depressed.append(st.lead)
    # STEMI requires CONTIGUOUS elevation — >=2 leads within one coronary
    # territory. Scattered single-lead elevation is early-repol / noise, not
    # infarction; the old "any >=2 leads" gate was the main false-positive source
    # (spec 0.64 -> 0.75, D15j).
    terr_hits = [leads for _, _, leads, _ in _TERRITORIES if len(set(elevated) & leads) >= 2]
    if terr_hits:
        # Diffuse ST elevation across >=2 coronary territories (and no single
        # reciprocal territory) points to pericarditis, not a focal STEMI — a
        # classic mimic distinction (ecglibrary / 150 ECG Problems).
        n_terr = len(terr_hits)
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
