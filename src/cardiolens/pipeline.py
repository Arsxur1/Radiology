"""End-to-end pipeline orchestration (TZ §3 layer flow).

Runs Layer 1 -> Layer 6 plus the cross-cutting safety scan, and assembles a
``PipelineResult`` carrying every intermediate artifact plus an audit log for
traceability (TZ §11.1, §12). Output rendering (JSON/SVG/graph/PDF) lives in
``cardiolens.outputs`` and consumes this result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
from cardiolens.contracts.findings import FindingSet
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

    # Layer 3 — measurements (TZ §6).
    m = measure(sig)
    audit.append(
        f"[L3] HR={m.heart_rate_bpm} PR={m.intervals.pr_ms} "
        f"QRS={m.intervals.qrs_ms} QTcB={m.intervals.qtc_bazett_ms}"
    )

    # Layer 4 — findings (TZ §7).
    norms = norms_for_patient(sig.patient.age_years, sig.patient.age_days)
    findings = detect_findings(sig, m, norms=norms, la_ra_reversed=la_ra_reversed)
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
