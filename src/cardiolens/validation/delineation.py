"""Delineation-accuracy validation against expert wave-boundary annotations.

The finding-level harness (``harness.py``) scores sensitivity/specificity of
diagnoses. This module measures the layer below it: how close Layer-3's
per-beat fiducials (P onset, QRS onset/offset, T offset) land to expert
annotations, reported as signed bias and MAE per interval.

It targets LUDB (Lobachevsky University Database, PhysioNet ``ludb/1.0.1``),
whose per-lead annotation files mark each wave as an onset ``(`` / peak
(``p`` P, ``N`` QRS, ``t`` T) / offset ``)`` triplet. Runs only against a local
checkout; PhysioNet is not required at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median


@dataclass
class IntervalAccuracy:
    name: str
    signed_errors: list[float] = field(default_factory=list)  # measured - gold (ms)

    def add(self, measured: float | None, gold: float | None) -> None:
        if measured is not None and gold is not None:
            self.signed_errors.append(measured - gold)

    @property
    def n(self) -> int:
        return len(self.signed_errors)

    @property
    def bias_ms(self) -> float | None:
        return round(median(self.signed_errors), 1) if self.signed_errors else None

    @property
    def mae_ms(self) -> float | None:
        return round(median(abs(e) for e in self.signed_errors), 1) if self.signed_errors else None

    def to_dict(self) -> dict:
        return {"name": self.name, "n": self.n, "bias_ms": self.bias_ms, "mae_ms": self.mae_ms}


@dataclass
class DelineationReport:
    n_records: int
    intervals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"n_records": self.n_records,
                "intervals": {k: v.to_dict() for k, v in self.intervals.items()}}


def _triplets(symbols, samples) -> dict[str, list[tuple[int, int, int]]]:
    """Group annotations into (onset, peak, offset) sample triplets per wave type."""
    out: dict[str, list[tuple[int, int, int]]] = {"p": [], "N": [], "t": []}
    i = 0
    while i < len(symbols):
        if symbols[i] == "(" and i + 2 < len(symbols) and symbols[i + 2] == ")":
            peak = symbols[i + 1]
            if peak in out:
                out[peak].append((int(samples[i]), int(samples[i + 1]), int(samples[i + 2])))
            i += 3
        else:
            i += 1
    return out


def _next_after(candidates: list[int], t: int) -> int | None:
    later = [c for c in candidates if c >= t]
    return min(later) if later else None


def gold_intervals_ms(symbols, samples, fs: float) -> tuple[float | None, float | None, float | None]:
    """Median gold PR / QRS / QT (ms) from LUDB-style boundary annotations.

    PR pairs each P onset with the next QRS onset; QT pairs each QRS onset with
    the next T offset; QRS is its own onset→offset. Medians across the annotated
    beats, so a single missed wave doesn't distort the record's value.
    """
    t = _triplets(symbols, samples)
    qrs = [(on, off) for on, _, off in t["N"]]
    qrs_onsets = [on for on, _ in qrs]
    t_offsets = [off for _, _, off in t["t"]]

    pr = [(nxt - p_on) for p_on, _, _ in t["p"]
          if (nxt := _next_after(qrs_onsets, p_on)) is not None]
    qt = [(nxt - on) for on, _ in qrs
          if (nxt := _next_after(t_offsets, on)) is not None]
    qrs_ms = [off - on for on, off in qrs]

    def med_ms(samples_list: list[int]) -> float | None:
        return round(median(samples_list) / fs * 1000.0, 1) if samples_list else None

    return med_ms(pr), med_ms(qrs_ms), med_ms(qt)


def validate_ludb_delineation(root: str | Path, limit: int = 50) -> DelineationReport:  # pragma: no cover
    """Compare Layer-3 intervals to LUDB expert annotations (lead II).

    Expects LUDB records under ``<root>/data/`` (``N.hea`` + ``N.dat`` + per-lead
    annotation files such as ``N.ii``). Runs only against a local checkout.
    """
    import wfdb

    from cardiolens.layer1_ingestion import load_wfdb
    from cardiolens.layer6_interpret import StubProvider
    from cardiolens.pipeline import run_pipeline

    root = Path(root)
    data = root / "data" if (root / "data").is_dir() else root
    recs = sorted({p.with_suffix("") for p in data.glob("*.hea")},
                  key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)[:limit]

    pr, qrs, qt = IntervalAccuracy("PR"), IntervalAccuracy("QRS"), IntervalAccuracy("QT")
    n = 0
    for rec in recs:
        try:
            ann = wfdb.rdann(str(rec), "ii")
            g_pr, g_qrs, g_qt = gold_intervals_ms(ann.symbol, ann.sample, ann.fs)
            result = run_pipeline(load_wfdb(rec), mode="A", provider=StubProvider())
        except Exception:  # noqa: S112 - skip unreadable records, keep validating the rest
            continue
        iv = result.measurements.intervals
        pr.add(iv.pr_ms, g_pr)
        qrs.add(iv.qrs_ms, g_qrs)
        qt.add(iv.qt_ms, g_qt)
        n += 1

    return DelineationReport(n_records=n,
                             intervals={"PR": pr, "QRS": qrs, "QT": qt})
