"""Ventricular hypertrophy (TZ §7.4): Sokolow-Lyon, Cornell, Romhilt-Estes; pediatric
age-band voltage norms (open item 2).

Adults: any of Sokolow-Lyon (SV1 + max(RV5, RV6)), Cornell (RaVL + SV3, sex-specific)
or Romhilt-Estes >= 5 points -> ``lvh``; Romhilt-Estes 4 points -> ``lvh_probable``
(non-critical). Children: the adult voltage criteria do not apply (tall R waves are
normal); LVH / RVH are called against the age band's 98th-percentile voltages
(Davignon 1979/80 as reproduced in pediatric ECG references, approximate), and only
with a supporting sign for LVH (strain or left axis) to stay conservative.
"""

from __future__ import annotations

from typing import Any

from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet


def _ev(**kv: Any) -> list[Evidence]:
    return [Evidence(feature=k, value=str(v)) for k, v in kv.items()]


def _strain(m: Measurements, depression_uv: float = 100.0) -> bool:
    """Strain pattern: T inversion AND >= 1 mm ST depression in V5/V6 with a positive
    QRS. Either sign alone fired on a quarter of PTB-XL's non-LVH records."""
    inv = set(m.morphology.inverted_t_leads)
    depressed = {st.lead for st in m.morphology.st
                 if st.lead in ("V5", "V6") and st.j60_uv is not None
                 and st.j60_uv <= -depression_uv and (st.qrs_net_uv or 0) > 0}
    return bool(({"V5", "V6"} & inv) and depressed)


def romhilt_estes_points(m: Measurements, norms: dict, fs: FindingSet) -> tuple[int, dict]:
    v = m.voltages
    re = norms.get("voltage_uv", {}).get("romhilt_estes", {})
    pts, why = 0, {}
    big = re.get("voltage_uv", 2000)
    if (v.r_max_limb_uv or 0) >= big or max(v.s_v1_uv or 0, v.s_v2_uv or 0) >= 1.5 * big \
            or max(v.r_v5_uv or 0, v.r_v6_uv or 0) >= 1.5 * big:
        pts += 3; why["voltage"] = 3
    if _strain(m, re.get("strain_depression_uv", 100)):
        pts += 3; why["strain"] = 3
    if (v.p_terminal_v1_uv_ms or 0) >= re.get("p_terminal_v1_uv_ms", 4000):
        pts += 3; why["left_atrial"] = 3
    if "left_axis_deviation" in fs.codes():
        pts += 2; why["left_axis"] = 2
    if (m.intervals.qrs_ms or 0) >= re.get("qrs_ms", 90):
        pts += 1; why["qrs_duration"] = 1
    if (v.intrinsicoid_v5_v6_ms or 0) >= re.get("intrinsicoid_ms", 50):
        pts += 1; why["intrinsicoid"] = 1
    return pts, why


def _pediatric_band(norms: dict, patient) -> dict | None:
    bands = norms.get("voltage_uv", {}).get("pediatric_98th_by_age") or []
    age_days, age_years = getattr(patient, "age_days", None), getattr(patient, "age_years", None)
    for band in bands:
        if "age_days_max" in band and age_days is not None and age_days <= band["age_days_max"]:
            return band
        if "age_years_max" in band and age_years is not None and age_years <= band["age_years_max"]:
            return band
    return None


_VOLTAGE_INVALID_CONTEXT = {"lbbb", "rbbb", "wpw_preexcitation", "wide_qrs_nonspecific"}


def hypertrophy_findings(m: Measurements, norms: dict, fs: FindingSet, patient=None) -> None:
    vt = norms.get("voltage_uv", {})
    v = m.voltages
    # Voltage criteria are not interpretable with a bundle branch block or
    # pre-excitation (the wide, notched QRS carries its own deep S / tall R): on
    # PTB-XL, Cornell fired on most LBBB records. Report nothing rather than guess.
    if fs.codes() & _VOLTAGE_INVALID_CONTEXT:
        return
    if vt.get("pediatric_98th_by_age"):
        band = _pediatric_band(norms, patient)
        if band is None:
            return
        lvh_v = [f"RV6 {v.r_v6_uv}>{band['r_v6']}" if (v.r_v6_uv or 0) > band["r_v6"] else None,
                 f"SV1 {v.s_v1_uv}>{band['s_v1']}" if (v.s_v1_uv or 0) > band["s_v1"] else None]
        rvh_v = [f"RV1 {v.r_v1_uv}>{band['r_v1']}" if (v.r_v1_uv or 0) > band["r_v1"] else None,
                 f"SV6 {v.s_v6_uv}>{band['s_v6']}" if (v.s_v6_uv or 0) > band["s_v6"] else None]
        lvh_v, rvh_v = [x for x in lvh_v if x], [x for x in rvh_v if x]
        support = _strain(m) or "left_axis_deviation" in fs.codes()
        if lvh_v and (support or len(lvh_v) == 2):
            fs.findings.append(Finding(
                code="lvh", group="hypertrophy",
                label_ru=f"Гипертрофия ЛЖ (детские нормы: {', '.join(lvh_v)} мкВ)",
                label_uz=f"Chap qorincha gipertrofiyasi (bolalar normasi: {', '.join(lvh_v)} µV)",
                evidence=_ev(criteria=";".join(lvh_v), band=band.get("label", ""), support=support),
                confidence=0.65,
            ))
        if len(rvh_v) == 2 or (rvh_v and "right_axis_deviation" in fs.codes()):
            fs.findings.append(Finding(
                code="rvh", group="hypertrophy",
                label_ru=f"Гипертрофия ПЖ (детские нормы: {', '.join(rvh_v)} мкВ)",
                label_uz=f"O'ng qorincha gipertrofiyasi (bolalar normasi: {', '.join(rvh_v)} µV)",
                evidence=_ev(criteria=";".join(rvh_v), band=band.get("label", "")), confidence=0.6,
            ))
        return

    fired, ev = [], {}
    sl_thr = vt.get("sokolow_lyon_lvh")
    if m.sokolow_lyon_uv is not None and sl_thr and m.sokolow_lyon_uv > sl_thr:
        fired.append("Sokolow-Lyon"); ev["sokolow_lyon_uv"] = m.sokolow_lyon_uv
    sex = (getattr(patient, "sex", None) or "male").lower()
    c_thr = vt.get("cornell_lvh_female" if sex == "female" else "cornell_lvh_male")
    if c_thr and v.r_avl_uv is not None and v.s_v3_uv is not None:
        cornell = v.r_avl_uv + v.s_v3_uv
        ev["cornell_uv"] = round(cornell, 1)
        if cornell > c_thr:
            fired.append("Cornell")
    pts, why = romhilt_estes_points(m, norms, fs)
    ev["romhilt_estes_points"] = pts
    re_thr = vt.get("romhilt_estes", {}).get("lvh_points", 5)
    # The voltage criteria (published thresholds) carry the `lvh` code; the point
    # score alone is reported as `lvh_probable` — on PTB-XL it adds 12 points of
    # sensitivity for 8 points of specificity, too loose for the main code (D39).
    if fired:
        fs.findings.append(Finding(
            code="lvh", group="hypertrophy",
            label_ru=f"Гипертрофия ЛЖ ({', '.join(fired)}" + (f"; Romhilt-Estes {pts} б.)" if pts >= re_thr else ")"),
            label_uz=f"Chap qorincha gipertrofiyasi ({', '.join(fired)}" + (f"; Romhilt-Estes {pts} ball)" if pts >= re_thr else ")"),
            evidence=_ev(criteria=",".join(fired), **ev, **{f"re_{k}": val for k, val in why.items()}),
            confidence=0.8 if (len(fired) >= 2 or pts >= re_thr) else 0.7,
        ))
    elif pts >= re_thr:
        fs.findings.append(Finding(
            code="lvh_probable", group="hypertrophy",
            label_ru=f"Вероятная гипертрофия ЛЖ (Romhilt-Estes {pts} балла, вольтажные критерии не достигнуты)",
            label_uz=f"Ehtimoliy chap qorincha gipertrofiyasi (Romhilt-Estes {pts} ball, voltaj mezonlari yo'q)",
            evidence=_ev(**ev, **{f"re_{k}": val for k, val in why.items()}), confidence=0.5,
        ))
