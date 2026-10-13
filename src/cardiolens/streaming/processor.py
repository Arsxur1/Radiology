"""Streaming pipeline processor (TZ §4).

Feeds overlapping windows through the full CardioLens pipeline online. Each
window is compared serially against the previous one (TZ §8.4), so the processor
surfaces the moment a critical finding *newly appears* in the stream — the key
signal for continuous monitoring (e.g. an ST elevation developing mid-recording).
Core layers run locally (TZ §12), so streaming works offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cardiolens.config import AppConfig
from cardiolens.contracts.signal import Patient, PatientContext, SignalRepresentation
from cardiolens.layer6_interpret import StubProvider
from cardiolens.layer6_interpret.provider import LLMProvider
from cardiolens.pipeline import PipelineResult, run_pipeline
from cardiolens.streaming.buffer import SlidingWindower


@dataclass
class WindowResult:
    """Outcome for a single stream window."""

    t_start_s: float
    t_end_s: float
    result: PipelineResult
    new_critical: list[str] = field(default_factory=list)  # critical codes new vs prev window

    @property
    def finding_codes(self) -> list[str]:
        return [f.code for f in self.result.findings.findings]

    @property
    def critical_codes(self) -> list[str]:
        return [a.code for a in self.result.safety.alerts]


class StreamProcessor:
    """Online windowed ECG analysis (TZ §4)."""

    def __init__(
        self,
        lead_names: list[str],
        sampling_rate_hz: float,
        *,
        window_s: float = 10.0,
        hop_s: float = 5.0,
        mode: str | None = None,
        patient: Patient | None = None,
        context: PatientContext | None = None,
        provider: LLMProvider | None = None,
        config: AppConfig | None = None,
    ):
        self.lead_names = list(lead_names)
        self.fs = float(sampling_rate_hz)
        self.window_s = window_s
        self.mode = mode
        self.patient = patient or Patient()
        self.context = context or PatientContext()
        # Default to the offline stub so streaming never blocks on the network.
        self.provider = provider or StubProvider()
        self.config = config
        self._windower = SlidingWindower(len(lead_names), sampling_rate_hz,
                                         window_s=window_s, hop_s=hop_s)
        self._prev: PipelineResult | None = None

    def push(self, chunk: np.ndarray) -> list[WindowResult]:
        """Feed a ``[n_leads, k]`` chunk; return results for completed windows."""
        out: list[WindowResult] = []
        for start, window in self._windower.push(chunk):
            sig = SignalRepresentation(
                signal=window, sampling_rate_hz=self.fs, lead_names=self.lead_names,
                duration_s=self.window_s, source="stream",
                patient=self.patient, context=self.context,
            )
            result = run_pipeline(sig, mode=self.mode, provider=self.provider,
                                  config=self.config, prior=self._prev)
            prev_crit = {a.code for a in self._prev.safety.alerts} if self._prev else set()
            new_crit = sorted({a.code for a in result.safety.alerts} - prev_crit)
            out.append(WindowResult(
                t_start_s=round(start / self.fs, 3),
                t_end_s=round((start + self._windower.window_n) / self.fs, 3),
                result=result, new_critical=new_crit,
            ))
            self._prev = result
        return out

    def process_signal(self, sig: SignalRepresentation,
                       chunk_s: float = 2.0) -> list[WindowResult]:
        """Convenience: replay a whole recording through the stream in chunks."""
        self.patient = sig.patient
        self.context = sig.context
        n = int(round(chunk_s * self.fs))
        results: list[WindowResult] = []
        for i in range(0, sig.n_samples, n):
            results.extend(self.push(sig.signal[:, i:i + n]))
        return results
