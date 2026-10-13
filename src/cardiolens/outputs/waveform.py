"""Annotated ECG waveform rendering (TZ §11.3).

Draws the classic ECG grid (small 1 mm / large 5 mm boxes, 25 mm/s, 10 mm/mV),
plots selected leads, and overlays R-peak markers and finding labels. Renders to
SVG (vector, for the report) or PNG.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cardiolens.pipeline import PipelineResult

# Standard scale: 25 mm/s, 10 mm/mV. One large box = 0.2 s x 0.5 mV.
_MM_PER_S = 25.0
_MM_PER_MV = 10.0


def render_waveform(
    result: PipelineResult,
    path: str,
    *,
    leads: list[str] | None = None,
    seconds: float = 5.0,
) -> str:
    sig = result.signal
    fs = sig.sampling_rate_hz
    leads = leads or [ln for ln in ("I", "II", "V1", "V5") if ln in sig.lead_names]
    n = int(min(seconds * fs, sig.n_samples))
    t = [i / fs for i in range(n)]

    fig, axes = plt.subplots(len(leads), 1, figsize=(10, 2.0 * len(leads)),
                             sharex=True)
    if len(leads) == 1:
        axes = [axes]

    r_peaks = [p for p in result.measurements.r_peaks_samples if p < n]

    for ax, lead in zip(axes, leads):
        row_mv = sig.lead(lead)[:n] / 1000.0  # µV -> mV for display
        ax.plot(t, row_mv, color="#b00020", linewidth=0.8)
        # ECG grid.
        ax.grid(which="major", color="#f4a6a6", linewidth=0.6)
        ax.grid(which="minor", color="#fbe0e0", linewidth=0.3)
        ax.set_xticks([x * 0.2 for x in range(int(seconds / 0.2) + 1)])
        ax.minorticks_on()
        ax.set_ylabel(f"{lead}\n(mV)", rotation=0, ha="right", va="center", fontsize=8)
        # R-peak markers.
        for rp in r_peaks:
            ax.axvline(rp / fs, color="#1565c0", alpha=0.25, linewidth=0.6)

    axes[-1].set_xlabel("Время, с (25 мм/с, 10 мм/мВ)", fontsize=8)
    finding_labels = ", ".join(f.code for f in result.findings.findings) or "—"
    fig.suptitle(f"ЭКГ — находки: {finding_labels}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, format="svg" if path.endswith(".svg") else None, dpi=120)
    plt.close(fig)
    return path
