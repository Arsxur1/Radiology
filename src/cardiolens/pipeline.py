"""End-to-end pipeline orchestration (TZ §3 layer flow).

Runs Layer 1 -> Layer 6 plus the cross-cutting safety scan, and assembles a
``PipelineResult`` carrying every intermediate artifact plus an audit log for
traceability (TZ §11.1, §12). Output rendering (JSON/SVG/graph/PDF) lives in
``cardiolens.outputs`` and consumes this result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cardiolens.arrhythmia import BeatClassifier, ectopy_findings, summarize_ectopy
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
from cardiolens.layer4_findings import detect_findings
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
    # Per-beat ventricular ectopy (TZ §7.1) — needs beat-level morphology, so it
    # rides the beat classifier rather than a single median interval. No-op until
    # a model is trained into configs/beat_lda.json.
    beat_model = BeatClassifier.load(configs_dir() / "beat_lda.json")
    if beat_model.fitted and m.r_peaks_samples:
        lead_idx = sig.lead_names.index("II") if "II" in sig.lead_names else 0
        summary = summarize_ectopy(sig.signal[lead_idx], m.r_peaks_samples,
                                   sig.sampling_rate_hz, beat_model)
        findings.findings.extend(ectopy_findings(summary))
        audit.append(f"[L4] ectopy: {summary.n_ventricular}/{summary.n_beats} VEB "
                     f"burden={summary.burden}")
        # Ventricular bigeminy makes RR irregularly-irregular and mimics AF on RR
        # alone. When bigeminy is detected, the irregularity is explained by the
        # ectopy, so suppress a co-emitted atrial_fibrillation (§7.1).
        if summary.bigeminy:
            af = findings.by_code("atrial_fibrillation")
            if af is not None:
                findings.findings.remove(af)
                audit.append("[L4] suppressed atrial_fibrillation (RR irregularity "
                             "explained by ventricular bigeminy)")
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
    safety = scan_critical(findings, m)
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


def _quality_ok(quality, threshold: float) -> bool:
    sqi = quality.per_lead_sqi
    if not sqi:
        return False
    good = sum(1 for s in sqi if s >= threshold)
    return good >= 0.7 * len(sqi)  # >=70% of leads acceptable
