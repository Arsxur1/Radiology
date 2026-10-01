"""End-to-end pipeline orchestration (TZ §3 layer flow).

Runs Layer 1 -> Layer 6 plus the cross-cutting safety scan, and assembles a
``PipelineResult`` carrying every intermediate artifact plus an audit log for
traceability (TZ §11.1, §12). Output rendering (JSON/SVG/graph/PDF) lives in
``cardiolens.outputs`` and consumes this result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cardiolens.arrhythmia import (
    detect_pvc_beats,
    detect_ventricular_runs,
    pvc_findings,
    vt_run_finding,
)
from cardiolens.arrhythmia.atrial import (
    AF_HEDGE_MIN,
    FLUTTER_2TO1_THRESHOLD,
    FLUTTER_HEDGE_MIN,
    P_AMPLITUDE_MAX,
    P_CONSISTENCY_MAX,
    atrial_organisation,
    flutter_2to1_score,
    p_window_amplitude,
    pre_qrs_consistency,
    two_to_one_features,
)
from cardiolens.calibration import TemperatureScaler
from cardiolens.config import (
    AppConfig,
    configs_dir,
    load_app_config,
    load_rule_base,
    norms_for_patient,
)
from cardiolens.contracts.causal import CausalGraph
from cardiolens.contracts.features import Measurements
from cardiolens.contracts.findings import Evidence, Finding, FindingSet
from cardiolens.contracts.report import InterpretationReport
from cardiolens.contracts.signal import SignalRepresentation
from cardiolens.layer2_preprocess import assess_quality, preprocess
from cardiolens.layer3_measure import measure
from cardiolens.layer3_measure.measure import (
    pacing_spike_coupling,
    pacing_spike_ratio,
    pacing_spike_times,
)
from cardiolens.layer4_findings import detect_findings
from cardiolens.layer4_findings.rules import _AFIB_RR_CV_MIN, _AFIB_RR_MAD_CV_MIN
from cardiolens.layer5_causal import analyze
from cardiolens.layer6_interpret import InterpretationInput, interpret
from cardiolens.layer6_interpret.provider import LLMProvider
from cardiolens.safety import SafetyResult, scan_critical
from cardiolens.safety.critical import CriticalAlert
from cardiolens.serial import SerialComparison, compare


@dataclass
class PipelineResult:
    signal: SignalRepresentation
    measurements: Measurements
    findings: FindingSet
    causal_graph: CausalGraph
    safety: SafetyResult
    report: InterpretationReport
    config: AppConfig
    audit_log: list[str] = field(default_factory=list)
    serial: SerialComparison | None = None


def run_pipeline(
    sig: SignalRepresentation,
    *,
    mode: str | None = None,
    provider: LLMProvider | None = None,
    config: AppConfig | None = None,
    calibrator: TemperatureScaler | None = None,
    prior: PipelineResult | None = None,
) -> PipelineResult:
    """Run the full CardioLens pipeline on one signal."""
    cfg = config or load_app_config()
    rule_base = load_rule_base()
    # Confidence calibration (TZ §13): identity until a scaler is fitted/saved.
    if calibrator is None:
        calibrator = TemperatureScaler.load(configs_dir() / "calibrator.json")
    audit: list[str] = []

    # Resting-ECG interpreter: a long recording is analysed on its leading window.
    max_s = cfg.preprocessing.max_analysis_s
    if max_s and sig.duration_s > max_s:
        n = int(round(max_s * sig.sampling_rate_hz))
        audit.append(f"[L1] recording {sig.duration_s:.0f} s > {max_s:.0f} s: "
                     f"analysed the first {max_s:.0f} s only")
        sig = sig.model_copy(update={"signal": sig.signal[:, :n], "duration_s": n / sig.sampling_rate_hz})

    # Pacing-spike detection runs on the RAW signal (§7.1): the narrow, high-slew
    # pacer spike is removed by the Layer-2 bandpass, so measure it before that.
    pacing_ratio = pacing_spike_ratio(sig)
    spike_times = pacing_spike_times(sig) if pacing_ratio >= cfg.pacing.spike_ratio_min else []

    # Layer 2 — preprocess + quality (TZ §5).
    pp = cfg.preprocessing
    sig = preprocess(
        sig,
        mains_frequency_hz=cfg.mains_frequency_hz,
        bandpass_low_hz=pp.bandpass_low_hz,
        bandpass_high_hz=pp.bandpass_high_hz,
        target_sampling_rate_hz=pp.target_sampling_rate_hz,
    )
    quality, la_ra_reversed = assess_quality(sig, sqi_accept_threshold=pp.sqi_accept_threshold)
    quality_ok = _quality_ok(quality, pp.sqi_accept_threshold)
    audit.append(f"[L2] SQI mean-ok={quality_ok}, flags={quality.flags}")
    # A trace digitised from a degraded image carries low provenance confidence
    # (§4, §0): interpret it, but never autonomously as if it were a clean recording.
    prov = sig.provenance
    if prov.digitized and prov.confidence < cfg.quality.digitized_confidence_min:
        quality_ok = False
        audit.append(f"[L1] digitised image, provenance confidence {prov.confidence} "
                     f"< {cfg.quality.digitized_confidence_min} — escalate")

    # Layer 3 — measurements (TZ §6), re-centred by the configured fiducial-bias
    # correction (DECISIONS.md D15a).
    m = measure(sig, correction=cfg.delineation_correction)
    audit.append(
        f"[L3] HR={m.heart_rate_bpm} PR={m.intervals.pr_ms} "
        f"QRS={m.intervals.qrs_ms} QTcB={m.intervals.qtc_bazett_ms}"
    )

    # Layer 4 — findings (TZ §7).
    norms = norms_for_patient(sig.patient.age_years, sig.patient.age_days)
    findings = detect_findings(sig, m, norms=norms, la_ra_reversed=la_ra_reversed)
    # Ventricular ectopy (TZ §7.1). VT (a §10 alert): runs of premature, abnormally
    # shaped beats on the 12-lead strip (D40); sustained fast VT is also caught by the
    # wide-complex tachycardia rule (D37). The MIT-BIH beat-classifier VT path was
    # removed (D41): 0 of 153 development V runs, and every one of its 12-lead calls
    # was a wide sinus/AF/SVT rhythm or pacing. Isolated PVCs and bigeminy use the
    # 12-lead PVC rule (D25).
    vt_called = False
    runs = detect_ventricular_runs(sig, m.r_peaks_samples or [], sig.sampling_rate_hz)
    if runs:
        findings.findings.append(vt_run_finding(runs))
        vt_called = True
        audit.append(f"[L4] ventricular run(s) (first beat, beats, rate): {runs}")
    pvc_idx = detect_pvc_beats(sig, m.r_peaks_samples or [], sig.sampling_rate_hz)
    if pvc_idx and not vt_called:
        findings.findings.extend(pvc_findings(pvc_idx, len(m.r_peaks_samples)))
        audit.append(f"[L4] 12-lead PVC rule: {len(pvc_idx)} PVC beat(s)")
    # PVCs make RR irregular and mimic AF. If the rhythm no longer meets the AF
    # criteria once the intervals around each PVC are removed, the irregularity
    # is explained by the ectopy, so AF is withheld (D25).
    if pvc_idx and findings.by_code("atrial_fibrillation") is not None:
        explained, note = _pvc_explains_irregularity(m.r_peaks_samples, pvc_idx)
        if explained:
            findings.findings.remove(findings.by_code("atrial_fibrillation"))
            audit.append(f"[L4] withheld atrial_fibrillation: RR irregularity explained "
                         f"by {len(pvc_idx)} PVC(s) ({note})")
    # Pause / dropped beat (TZ §7.1, §10) — R-peak based, as reliable as HR.
    pc = cfg.pause
    if m.max_rr_ms is not None and (m.max_rr_ms >= pc.finding_ms
                                    or (m.pause_ratio or 0) >= pc.ratio):
        findings.findings.append(Finding(
            code="sinus_pause", group="rhythm",
            label_ru=f"Пауза ритма {m.max_rr_ms:.0f} мс (RR ×{m.pause_ratio})",
            label_uz=f"Ritm pauzasi {m.max_rr_ms:.0f} ms (RR ×{m.pause_ratio})",
            evidence=[Evidence(feature="max_rr_ms", value=f"{m.max_rr_ms:.0f}"),
                      Evidence(feature="pause_ratio", value=str(m.pause_ratio))],
            confidence=0.8, critical=m.max_rr_ms >= pc.critical_ms))
        audit.append(f"[L4] pause: max_rr={m.max_rr_ms} ms ratio={m.pause_ratio}")
    # Paced rhythm (TZ §7.1) — the pacer spike is an extreme slope outlier on the
    # RAW signal (measured above, before the bandpass that would erase it).
    # The spike must also be time-locked to the QRS: an artifact spike is an
    # equally extreme slope outlier but lands at random (DECISIONS D24).
    coupling = pacing_spike_coupling(
        spike_times, [r / sig.sampling_rate_hz for r in (m.r_peaks_samples or [])])
    if pacing_ratio >= cfg.pacing.spike_ratio_min and coupling >= cfg.pacing.coupling_min:
        findings.findings.append(Finding(
            code="paced_rhythm", group="rhythm",
            label_ru=f"Ритм электрокардиостимулятора (спайк ×{pacing_ratio}, "
                     f"связан с QRS в {coupling:.0%} ударов)",
            label_uz=f"Kardiostimulyator ritmi (spike ×{pacing_ratio}, QRS bilan {coupling:.0%})",
            evidence=[Evidence(feature="pacing_spike_ratio", value=str(pacing_ratio)),
                      Evidence(feature="spike_qrs_coupling", value=str(coupling))],
            confidence=0.8))
        audit.append(f"[L4] paced rhythm: spike_ratio={pacing_ratio} coupling={coupling}")
    elif pacing_ratio >= cfg.pacing.spike_ratio_min:
        audit.append(f"[L4] slope outlier ×{pacing_ratio} not QRS-locked "
                     f"(coupling {coupling}) — not called paced")
    _apply_rhythm_hierarchy(findings, m, audit)
    _check_sinus_p_waves(findings, sig, m, audit)
    _flutter_2to1_hedge(findings, sig, m, audit)
    _af_atrial_evidence(findings, sig, m, audit)
    # Signal-organization gate (§5) — runs AFTER all finding emission. If the QRS
    # complexes are not self-consistent, the R-peaks (hence HR, intervals, rhythm)
    # are unreliable, so withhold every R-peak-derived finding and report the poor
    # quality instead of, e.g., a bradycardia read off noise. Real ECG (incl. AFib)
    # stays well above the gate. quality_ok is forced low so mode B also escalates.
    tc = m.qrs_template_consistency
    if tc is not None and tc < cfg.quality.beat_consistency_min:
        findings.findings = [Finding(
            code="poor_quality_signal", group="quality",
            label_ru=f"Сигнал непригоден для надёжной интерпретации "
                     f"(консистентность QRS {tc} < {cfg.quality.beat_consistency_min})",
            label_uz=f"Signal ishonchli talqin uchun yaroqsiz (QRS konsistentligi {tc})",
            evidence=[Evidence(feature="qrs_template_consistency", value=str(tc))],
            confidence=0.9)]
        quality_ok = False
        audit.append(f"[L2] signal disorganized (beat consistency {tc}) — "
                     f"R-peak findings withheld")
    # Apply confidence calibration to each finding (TZ §13).
    for f in findings.findings:
        f.confidence = round(calibrator.apply(f.confidence), 3)
    audit.append(
        f"[L4] findings={[f.code for f in findings.findings]} "
        f"(calibrated T={calibrator.temperature})"
    )

    # Serial comparison against a prior ECG (TZ §8.4).
    serial: SerialComparison | None = None
    if prior is not None:
        serial = compare(findings, m, prior.findings, prior.measurements,
                         prior_id=sig.context.prior_ecg_id)
        audit.append(f"[serial] deltas={[d.code for d in serial.deltas]} "
                     f"qtcΔ={serial.qtc_delta_ms}")

    # Safety scan (TZ §10) — before interpretation, independent of LLM.
    safety = scan_critical(findings, m,
                           long_qt_critical_ms=cfg.safety.long_qt_critical_qtc_ms,
                           patient=sig.patient, context=sig.context,
                           tall_t_precordial_uv=cfg.safety.tall_t_precordial_uv,
                           tall_t_limb_uv=cfg.safety.tall_t_limb_uv,
                           hyperkalaemia_potassium_mmol=cfg.safety.hyperkalaemia_potassium_mmol)
    # New critical dynamics on serial comparison force escalation (TZ §8.4, §10).
    if serial is not None:
        for d in serial.deltas:
            if d.critical:
                safety.alerts.append(CriticalAlert(
                    code=f"serial_{d.code}", label_ru=d.label_ru, label_uz=d.label_uz,
                    reason="significant serial change", log_id=f"serial-{d.code}"))
                safety.audit_log.append(f"[serial] CRITICAL new/changed {d.code}")
    audit.extend(safety.audit_log)

    # Layer 5 — causal analysis (TZ §8).
    causal = analyze(findings, rule_base, sig.context)
    # Serial dynamics strengthen confidence (TZ §8.4).
    if serial is not None and serial.has_significant_change:
        causal.confidence = round(min(1.0, causal.confidence + 0.05), 3)
    audit.append(
        f"[L5] rules={causal.activated_rules} critical={causal.critical_flags} "
        f"implausible={causal.implausible_flags} conf={causal.confidence}"
    )

    # Layer 6 — interpretation with mode policy (TZ §9, §2).
    requested_mode = mode or cfg.mode
    data = InterpretationInput(
        findings=findings, causal_graph=causal, measurements=m,
        patient=sig.patient, context=sig.context, mode=requested_mode,
        critical_labels=[(a.label_ru, a.label_uz) for a in safety.alerts],
        disclaimer_ru=cfg.disclaimers.ru, disclaimer_uz=cfg.disclaimers.uz,
    )
    report = interpret(
        data, safety, provider=provider,
        min_overall=cfg.confidence.autonomous_min_overall, quality_ok=quality_ok,
    )
    audit.append(f"[L6] provider={report.mode} escalation={report.escalation_reason}")

    return PipelineResult(
        signal=sig, measurements=m, findings=findings, causal_graph=causal,
        safety=safety, report=report, config=cfg, audit_log=audit, serial=serial,
    )


# Rhythm hierarchy (§7.1, DECISIONS D15z): in atrial fibrillation there is no
# sinus rhythm and no measurable PR, so these co-emitted labels contradict the AF
# call. Each suppression was kept only if it removed more false than true labels
# on development data (Chapman/Ningbo + PTB-XL outside fold 10). NOT suppressed:
# sinus_bradycardia (removed as many true as false labels — slow sinus
# arrhythmia still draws AF calls).
_AF_EXCLUDES = ("sinus_rhythm", "sinus_tachycardia", "first_degree_av_block",
                "irregular_rhythm")
# A paced QRS invalidates bundle-branch, axis, voltage and repolarization criteria
# and the PR interval (D24). Enabled only once the paced call was QRS-locked; on
# development data every suppression then removed only or mostly false labels
# (e.g. lbbb +26/-0, t_inversion +43/-2, first_degree_av_block +35/-5).
# st_elevation is deliberately NOT suppressed: it is a §10 must-not-miss finding
# and STEMI remains diagnosable in paced rhythm (Sgarbossa criteria).
# pathological_q_waves: paced QRS are QS-like in V1-V4, Q criteria do not apply (D36).
_PACED_EXCLUDES = ("lbbb", "rbbb", "incomplete_rbbb", "wide_qrs_nonspecific", "lvh", "first_degree_av_block",
                   "sinus_rhythm", "left_axis_deviation", "st_depression", "t_inversion",
                   "pathological_q_waves")


def _apply_rhythm_hierarchy(findings, m, audit: list[str]) -> None:
    if findings.by_code("paced_rhythm") is not None:
        dropped = [f.code for f in findings.findings if f.code in _PACED_EXCLUDES]
        findings.findings = [f for f in findings.findings if f.code not in _PACED_EXCLUDES]
        if dropped:
            audit.append(f"[L4] rhythm hierarchy: paced rhythm, withheld {dropped}")
    af = findings.by_code("atrial_fibrillation")
    if af is None:
        return
    dropped = [f.code for f in findings.findings if f.code in _AF_EXCLUDES]
    findings.findings = [f for f in findings.findings if f.code not in _AF_EXCLUDES]
    hr = m.heart_rate_bpm
    if hr is not None:
        # Ventricular-rate form of AF (the clinically actionable descriptor).
        if hr > 100:
            ru, uz = "тахисистолическая форма", "taxisistolik shakl"
        elif hr < 60:
            ru, uz = "брадисистолическая форма", "bradisistolik shakl"
        else:
            ru, uz = "нормосистолическая форма", "normosistolik shakl"
        af.label_ru = f"{af.label_ru}, {ru} (ЧЖС {hr:.0f}/мин)"
        af.label_uz = f"{af.label_uz}, {uz} (QQS {hr:.0f}/daq)"
        af.evidence.append(Evidence(feature="ventricular_rate_bpm", value=f"{hr:.0f}"))
    if dropped:
        audit.append(f"[L4] rhythm hierarchy: AF present, withheld {dropped}")


_SINUS_CODES = ("sinus_rhythm", "sinus_tachycardia", "sinus_bradycardia")


def _check_sinus_p_waves(findings, sig, m, audit: list[str]) -> None:
    """A rate-based sinus label needs an organised P wave before the QRS (D45).

    Flutter with fixed conduction, junctional and spikeless paced rhythms were
    labelled "sinus" from the rate alone. Without a repeating, visible pre-QRS
    wave in lead II the sinus label is replaced by a hedged rhythm finding.
    """
    if findings.by_code("atrial_fibrillation") is not None:
        return
    if not any(findings.by_code(c) is not None for c in _SINUS_CODES):
        return
    rpeaks = m.r_peaks_samples or []
    _, per_lead = pre_qrs_consistency(sig, rpeaks, m.intervals.qt_ms)
    consistency = per_lead.get("II")
    amplitude = p_window_amplitude(sig, rpeaks, "II")
    no_p = ((consistency is not None and consistency <= P_CONSISTENCY_MAX)
            or (amplitude is not None and amplitude <= P_AMPLITUDE_MAX))
    if not no_p:
        return
    if findings.by_code("sinus_rhythm") is not None:
        findings.findings.remove(findings.by_code("sinus_rhythm"))
    hr = m.heart_rate_bpm
    findings.findings.append(Finding(
        code="rhythm_without_p_waves", group="rhythm",
        label_ru=f"Ритм без чётких зубцов P (ЧСС {hr}/мин) — синусовый не подтверждён; "
                 f"исключить трепетание предсердий, узловой ритм, ЭКС",
        label_uz=f"Aniq P to'lqinlarsiz ritm (YUT {hr}/daq) — sinus tasdiqlanmagan; "
                 f"bo'lmachalar titrashi, tugun ritmi, EKS ni istisno qilish",
        evidence=[Evidence(feature="p_consistency_II", value=str(consistency)),
                  Evidence(feature="p_amplitude_ratio_II", value=str(amplitude))],
        confidence=0.5))
    audit.append(f"[L4] no organised P wave (II consistency {consistency}, amplitude "
                 f"{amplitude}) — sinus label withheld")


def _flutter_2to1_hedge(findings, sig, m, audit: list[str]) -> None:
    """A sinus label is hedged when lead II looks like 2:1 atrial flutter (D54).

    The label stays: the model flags ~2% of true sinus rhythm/tachycardia.
    """
    if findings.by_code("atrial_fibrillation") is not None or findings.by_code("paced_rhythm") is not None:
        return
    sinus = next((findings.by_code(c) for c in _SINUS_CODES if findings.by_code(c) is not None), None)
    if sinus is None:
        return
    rpeaks = m.r_peaks_samples or []
    _, per_lead = pre_qrs_consistency(sig, rpeaks, m.intervals.qt_ms)
    feats = two_to_one_features(sig, rpeaks)
    values = {"pc_II": per_lead.get("II"), "rr_cv": m.rr_cv, "flat_II": feats["flat"],
              "psign_II": feats["psign"], "h2_II": feats["h2"], "hr": m.heart_rate_bpm}
    score = flutter_2to1_score(values)
    if score <= FLUTTER_2TO1_THRESHOLD:
        return
    sinus.label_ru = (f"{sinus.label_ru}; возможно трепетание предсердий с проведением 2:1 — "
                      f"оценить волны F (II, III, aVF, V1)")
    sinus.label_uz = (f"{sinus.label_uz}; 2:1 o'tkazuvchanlikli bo'lmachalar titrashi bo'lishi mumkin — "
                      f"F to'lqinlarini baholash (II, III, aVF, V1)")
    sinus.evidence.append(Evidence(feature="flutter_2to1_score", value=str(score)))
    audit.append(f"[L4] sinus label hedged: 2:1 flutter score {score} > {FLUTTER_2TO1_THRESHOLD} "
                 f"({values})")


def _af_atrial_evidence(findings, sig, m, audit: list[str]) -> None:
    """Pre-QRS consistency as evidence on an AF call, hedged when P waves recur (D51).

    The AF call itself is never withdrawn here: withholding it on this feature cost
    3-5% AF sensitivity on held-out data (D42). A recurring pre-QRS wave in both II
    and V1 points to sinus rhythm with ectopy, so the label says so.
    """
    af = findings.by_code("atrial_fibrillation")
    if af is None:
        return
    _flutter_hedge(af, sig, m, audit)
    value, per_lead = pre_qrs_consistency(sig, m.r_peaks_samples or [], m.intervals.qt_ms)
    for lead, v in per_lead.items():
        af.evidence.append(Evidence(feature=f"p_consistency_{lead}", value=str(v)))
    if value is None or len(per_lead) < 2 or value < AF_HEDGE_MIN:
        return
    af.label_ru = (f"{af.label_ru}; перед QRS повторяется зубец P (II/V1 ≥ {value}) — "
                   f"исключить синусовый ритм с экстрасистолией")
    af.label_uz = (f"{af.label_uz}; QRS oldidan P to'lqini takrorlanadi (II/V1 ≥ {value}) — "
                   f"ekstrasistoliyali sinus ritmini istisno qilish")
    audit.append(f"[L4] AF call hedged: recurring pre-QRS wave {per_lead}")


def _flutter_hedge(af, sig, m, audit: list[str]) -> None:
    """An organised atrial residual hedges the AF label toward flutter (D53).

    The call stays AF: renaming it on this index renamed 7.5% of true AF on held-out
    data (D46).
    """
    organisation, dom_hz = atrial_organisation(sig, m.r_peaks_samples or [])
    if organisation is None:
        return
    rate = round(dom_hz * 60) if dom_hz else None
    af.evidence.append(Evidence(feature="atrial_organisation", value=str(organisation)))
    if organisation < FLUTTER_HEDGE_MIN:
        return
    af.evidence.append(Evidence(feature="atrial_rate_per_min", value=str(rate)))
    af.label_ru = (f"{af.label_ru}; организованная предсердная активность (~{rate}/мин) — "
                   f"исключить трепетание предсердий")
    af.label_uz = (f"{af.label_uz}; tartibli bo'lmacha faolligi (~{rate}/daq) — "
                   f"bo'lmachalar titrashini istisno qilish")
    audit.append(f"[L4] AF call hedged toward flutter: atrial organisation {organisation} "
                 f">= {FLUTTER_HEDGE_MIN} (~{rate}/min)")


def _pvc_explains_irregularity(r_peaks, pvc_idx) -> tuple[bool, str]:
    """True if, with the RR intervals around each PVC removed, the rhythm no
    longer meets the AF RR criteria (D25). Needs >= 4 remaining intervals."""
    rr = np.diff(np.asarray(r_peaks, dtype=float))
    drop = {j for i in pvc_idx for j in (i - 1, i)}
    clean = np.array([x for j, x in enumerate(rr) if j not in drop])
    if clean.size < 4:
        return False, "too few intervals left"
    med = float(np.median(clean))
    cv = float(np.std(clean) / np.mean(clean))
    mad_cv = 1.4826 * float(np.median(np.abs(clean - med))) / med
    explained = not (cv >= _AFIB_RR_CV_MIN and mad_cv >= _AFIB_RR_MAD_CV_MIN)
    return explained, f"clean CV {cv:.2f}, MAD-CV {mad_cv:.2f}"


def _quality_ok(quality, threshold: float) -> bool:
    sqi = quality.per_lead_sqi
    if not sqi:
        return False
    good = sum(1 for s in sqi if s >= threshold)
    return good >= 0.7 * len(sqi)  # >=70% of leads acceptable
