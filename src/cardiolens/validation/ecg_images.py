"""Paper-ECG photographs paired with their digital records (TZ §4, §16; blocker 3).

A folder holds images (``<name>.png`` / ``.jpg`` / ``.jpeg``, or the PhysioNet
Challenge-2024 / ECG-Image-Database layout ``<name>-0001.png``) and the WFDB
record each one was taken from (``<name>.hea`` + ``.dat``). Every image is
digitised with the page digitiser (self-calibrated: ``px_per_mm`` and
``mm_per_mV`` from the grid and the calibration pulse), run through the same
pipeline as the record, and compared with it on exactly the metrics of the
rendered-page benchmark: per-lead trace correlation, heart-rate / axis / QRS
differences and the overlap of the scored finding sets.

    python -m cardiolens.validate --ecg-images <folder>

No reachable PhysioNet source holds such photographs today (ECG-Image-Database and
the 2024 Challenge images return 404 from this environment); the adapter exists so
that a manually supplied set runs unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def pair_images_with_records(folder: str | Path) -> list[tuple[Path, Path]]:
    """(image, record-without-extension) pairs; ``name-0001.png`` pairs with ``name``."""
    folder = Path(folder)
    records = {p.stem: p.with_suffix("") for p in folder.rglob("*.hea")}
    pairs = []
    for img in sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES):
        stem = img.stem
        candidates = [stem, re.sub(r"-\d{4}$", "", stem)]
        rec = next((records[c] for c in candidates if c in records), None)
        if rec is not None and rec.with_suffix(".dat").exists():
            pairs.append((img, rec))
    return pairs


@dataclass
class ImageCase:
    name: str
    r_mean: float | None = None
    hr_delta: float | None = None
    axis_delta: float | None = None
    qrs_delta: float | None = None
    jaccard: float | None = None
    identical: bool | None = None
    error: str | None = None


@dataclass
class ImageReport:
    cases: list[ImageCase] = field(default_factory=list)

    @property
    def n_ok(self) -> int:
        return sum(1 for c in self.cases if c.error is None)

    def summary(self) -> dict:
        ok = [c for c in self.cases if c.error is None]

        def _m(vals, fn=np.mean):
            vals = [v for v in vals if v is not None]
            return round(float(fn(vals)), 3) if vals else None

        return {
            "n": len(self.cases), "digitised": len(ok), "failed": len(self.cases) - len(ok),
            "r_mean": _m([c.r_mean for c in ok]),
            "hr_mae": _m([c.hr_delta for c in ok]),
            "hr_within3": _m([c.hr_delta <= 3 for c in ok if c.hr_delta is not None]),
            "axis_mae": _m([c.axis_delta for c in ok]),
            "axis_within15": _m([c.axis_delta <= 15 for c in ok if c.axis_delta is not None]),
            "qrs_mae": _m([c.qrs_delta for c in ok]),
            "jaccard": _m([c.jaccard for c in ok]),
            "codes_identical": _m([c.identical for c in ok if c.identical is not None]),
        }

    def lines(self) -> list[str]:
        s = self.summary()
        out = [f"images {s['n']}, digitised {s['digitised']}, failed {s['failed']}"]
        if s["digitised"]:
            out.append(f"trace r {s['r_mean']} | HR MAE {s['hr_mae']} bpm (within 3: {s['hr_within3']}) | "
                       f"axis MAE {s['axis_mae']}° (within 15: {s['axis_within15']}) | QRS MAE {s['qrs_mae']} ms | "
                       f"finding Jaccard {s['jaccard']} | identical sets {s['codes_identical']}")
        for c in self.cases:
            if c.error:
                out.append(f"  {c.name}: FAILED ({c.error})")
        return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def evaluate_images(folder: str | Path, limit: int = 5000) -> ImageReport:  # pragma: no cover - needs images
    from cardiolens.layer1_ingestion import digitize as D
    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer6_interpret import StubProvider
    from cardiolens.pipeline import run_pipeline
    from cardiolens.validation.harness import SCORED_LABELS

    scored = set(SCORED_LABELS)
    report = ImageReport()
    for img, rec in pair_images_with_records(folder)[:limit]:
        case = ImageCase(name=img.name)
        report.cases.append(case)
        try:
            ref = run_pipeline(load_wfdb(rec), mode="A", provider=StubProvider())
            dig = D.digitize_page_to_signal(str(img), px_per_mm=None, mm_per_mv=None)
            traces = D.digitize_page(str(img), px_per_mm=None, mm_per_mv=None)
            res = run_pipeline(dig, mode="A", provider=StubProvider())
        except Exception as exc:
            case.error = f"{type(exc).__name__}: {exc}"[:120]
            continue
        psig = ref.signal
        seg = int(D.PANEL_SECONDS * psig.sampling_rate_hz)
        rs = []
        for leads in D.LAYOUT_3X4:
            for col, lead in enumerate(leads):
                if lead in traces and lead in psig.lead_names:
                    rs.append(_corr(psig.lead(lead)[col * seg:(col + 1) * seg], np.asarray(traces[lead], float)))
        case.r_mean = round(float(np.mean(rs)), 3) if rs else None
        rm, dm = ref.measurements, res.measurements
        if rm.heart_rate_bpm and dm.heart_rate_bpm:
            case.hr_delta = round(abs(rm.heart_rate_bpm - dm.heart_rate_bpm), 1)
        if rm.axes.qrs_deg is not None and dm.axes.qrs_deg is not None:
            d = abs(rm.axes.qrs_deg - dm.axes.qrs_deg)
            case.axis_delta = round(min(d, 360 - d), 1)
        if rm.intervals.qrs_ms and dm.intervals.qrs_ms:
            case.qrs_delta = round(abs(rm.intervals.qrs_ms - dm.intervals.qrs_ms), 1)
        a, b = ref.findings.codes() & scored, res.findings.codes() & scored
        case.jaccard = round(len(a & b) / len(a | b), 3) if (a | b) else 1.0
        case.identical = a == b
    return report
