"""Per-beat classification for ventricular ectopy (TZ §7.1).

Rate-based rhythm findings (brady/tachy, AF) ride reliable R-peaks; ectopic-beat
findings need per-beat morphology, which a single median interval can't express.
This module classifies each beat as N (normal), S (supraventricular ectopic) or
V (ventricular ectopic) from scale-invariant features, using a transparent LDA
whose coefficients live in ``configs/beat_lda.json`` (trained offline on MIT-BIH;
see ``cardiolens.train_beats``). Like the confidence calibrator, it degrades to a
no-op when the model file is absent.

Only ventricular ectopy (V) is reported: on an inter-patient MIT-BIH split it
reaches ~0.84 sensitivity / 0.99 PPV, whereas the supraventricular class is not
reliably separable from normal with these features (documented in DECISIONS.md).
Features are scale/rate-invariant (RR ratios, QRS width in ms, amplitude relative
to the record's own beats, morphology correlation to the dominant beat) so a model
trained at 360 Hz in mV transfers to the pipeline's 500 Hz µV lead II.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# QRS half-window for morphology features (~70 ms each side).
_HALF_MS = 70.0
# Number of normalized QRS-shape samples (scale/rate-invariant morphology).
_SHAPE_K = 10

FEATURES = (["pre_rr_ratio", "post_rr_ratio", "local_rr_ms",
             "qrs_width_ms", "amp_ratio", "template_corr", "prematurity", "comp_ratio"]
            + [f"shape_{i}" for i in range(_SHAPE_K)])


def extract_beat_features(lead: np.ndarray, rpeaks, fs: float) -> tuple[np.ndarray, list[int]]:
    """Scale-invariant per-beat features from one (filtered) lead in µV.

    Returns the feature matrix and the beat indices it covers (interior beats
    with a full QRS window). The morphology template is the median of ALL beat
    windows — label-free, so runtime matches training. See module docstring.
    """
    lead = np.asarray(lead, dtype=float)
    rp = np.asarray(rpeaks, dtype=int)
    n = len(rp)
    half = int(_HALF_MS / 1000.0 * fs)
    if n < 4 or half < 3:
        return np.empty((0, len(FEATURES))), []

    rr = np.diff(rp) / fs * 1000.0  # ms, length n-1
    segs = [lead[r - half:r + half] for r in rp if r - half >= 0 and r + half < len(lead)]
    if len(segs) < 10:
        return np.empty((0, len(FEATURES))), []
    tmpl = np.median(np.vstack(segs), axis=0)
    tmpl_amp = float(np.max(tmpl) - np.min(tmpl)) or 1.0

    rows, idx = [], []
    for i in range(1, n - 1):
        r = rp[i]
        if r - half < 0 or r + half >= len(lead):
            continue
        pre, post = rr[i - 1], rr[i]
        lo = max(0, i - 6)
        loc = float(np.median(rr[lo:i + 6])) if i + 6 <= len(rr) else float(np.median(rr))
        loc = loc or 1.0
        seg = lead[r - half:r + half]
        base = np.median(seg)
        amp = (float(np.max(seg) - np.min(seg))) / (tmpl_amp + 1e-9)
        thr = 0.2 * np.max(np.abs(seg - base))
        above = np.where(np.abs(seg - base) >= thr)[0]
        width = (above[-1] - above[0]) / fs * 1000.0 if len(above) >= 2 else 0.0
        denom = (np.std(seg) * np.std(tmpl))
        corr = float(np.mean((seg - seg.mean()) * (tmpl - tmpl.mean())) / denom) if denom else 0.0
        # Normalized QRS shape (baseline-removed, unit-amplitude, resampled to K):
        # scale/rate-invariant morphology that separates wide ectopic beats.
        norm = (seg - base)
        peak = np.max(np.abs(norm)) or 1.0
        shape = np.interp(np.linspace(0, len(norm)-1, _SHAPE_K),
                          np.arange(len(norm)), norm / peak)
        rows.append([pre / loc, post / loc, loc, width, amp, corr,
                     (loc - pre) / loc, post / loc, *shape.tolist()])
        idx.append(i)
    return np.asarray(rows), idx


@dataclass
class BeatClassifier:
    """Transparent LDA over beat features; loads coefficients from JSON."""

    classes: list[str] = field(default_factory=list)
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None
    means: dict[str, np.ndarray] = field(default_factory=dict)
    inv_cov: np.ndarray | None = None
    log_priors: dict[str, float] = field(default_factory=dict)
    fitted: bool = False

    @classmethod
    def load(cls, path: str | Path) -> BeatClassifier:
        path = Path(path)
        if not path.exists():
            return cls()  # no-op until a model is trained (mirrors the calibrator)
        d = json.loads(path.read_text())
        import math
        return cls(
            classes=list(d["classes"]),
            mu=np.asarray(d["mu"], float), sd=np.asarray(d["sd"], float),
            means={c: np.asarray(v, float) for c, v in d["means"].items()},
            inv_cov=np.asarray(d["inv_cov"], float),
            log_priors={c: math.log(p) for c, p in d["priors"].items()},
            fitted=True,
        )

    def predict(self, X: np.ndarray) -> list[str]:
        if not self.fitted or len(X) == 0:
            return ["N"] * len(X)
        Xz = (X - self.mu) / self.sd
        out = []
        for x in Xz:
            best, bs = "N", -np.inf
            for c in self.classes:
                d = x - self.means[c]
                score = -0.5 * d @ self.inv_cov @ d + self.log_priors[c]
                if score > bs:
                    bs, best = score, c
            out.append(best)
        return out


def ectopy_findings(summary: EctopySummary):
    """Build ventricular-ectopy findings from a beat-classification summary (§7.1).

    Reported only when ≥2 ventricular beats are classified (a single beat is
    within the per-beat error), so an incidental misclassification on a short
    strip doesn't raise a finding.
    """
    from cardiolens.contracts.findings import Evidence, Finding

    out = []
    if summary.n_ventricular < 2:
        return out
    ev = [Evidence(feature="pvc_count", value=str(summary.n_ventricular)),
          Evidence(feature="beats", value=str(summary.n_beats)),
          Evidence(feature="burden", value=f"{summary.burden:.0%}")]
    # A run of >=3 consecutive ventricular beats is ventricular tachycardia
    # (non-sustained salvo) — a §10 must-not-miss; takes precedence.
    if summary.max_vt_run >= 3:
        out.append(Finding(
            code="ventricular_tachycardia", group="rhythm",
            label_ru=f"Желудочковая тахикардия (пробежка {summary.max_vt_run} ЖЭ подряд)",
            label_uz=f"Ventrikulyar taxikardiya ({summary.max_vt_run} ta ketma-ket VES)",
            evidence=[*ev, Evidence(feature="run", value=str(summary.max_vt_run))],
            confidence=0.8, critical=True,
        ))
        return out
    if summary.bigeminy:
        out.append(Finding(
            code="ventricular_bigeminy", group="rhythm",
            label_ru=f"Желудочковая бигеминия (ЖЭ {summary.burden:.0%} ударов)",
            label_uz=f"Ventrikulyar bigeminiya (VES {summary.burden:.0%})",
            evidence=ev, confidence=0.7,
        ))
    else:
        out.append(Finding(
            code="ventricular_ectopy", group="rhythm",
            label_ru=f"Желудочковая экстрасистолия ({summary.n_ventricular} ЖЭ / "
                     f"{summary.n_beats} уд., {summary.burden:.0%})",
            label_uz=f"Ventrikulyar ekstrasistoliya ({summary.n_ventricular} VES / "
                     f"{summary.n_beats}, {summary.burden:.0%})",
            evidence=ev, confidence=0.75,
        ))
    return out


@dataclass
class EctopySummary:
    n_beats: int = 0
    n_ventricular: int = 0
    bigeminy: bool = False
    max_vt_run: int = 0  # longest run of consecutive ventricular beats

    @property
    def burden(self) -> float:
        return round(self.n_ventricular / self.n_beats, 3) if self.n_beats else 0.0


def summarize_ectopy(lead: np.ndarray, rpeaks, fs: float, model: BeatClassifier) -> EctopySummary:
    """Classify beats and summarise ventricular ectopy (count, burden, bigeminy)."""
    X, idx = extract_beat_features(lead, rpeaks, fs)
    if len(X) == 0 or not model.fitted:
        return EctopySummary(n_beats=len(idx))
    labels = model.predict(X)
    is_v = [lab == "V" for lab in labels]
    n_v = sum(is_v)
    # Longest run of consecutive ventricular beats (>=3 -> VT/NSVT salvo).
    max_run = run = 0
    for v in is_v:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    # Bigeminy: PVC roughly every other beat (alternating N-V-N-V), so ~half the
    # classified beats are V and they mostly alternate with normals.
    from itertools import pairwise
    alt = sum(1 for a, b in pairwise(is_v) if a != b)
    bigeminy = n_v >= 4 and 0.35 <= (n_v / len(is_v)) <= 0.6 and alt >= 0.6 * len(is_v)
    return EctopySummary(n_beats=len(is_v), n_ventricular=n_v,
                         bigeminy=bigeminy, max_vt_run=max_run)
