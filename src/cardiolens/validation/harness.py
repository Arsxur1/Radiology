"""Validation harness (TZ §15, §16).

Data-source agnostic: given labelled cases (signal + gold findings + gold
intervals), it runs the full pipeline and computes per-class sensitivity/
specificity and interval MAE (TZ §16). Two dataset adapters ship:

  * ``build_synthetic_dataset`` — deterministic, offline, ground truth known
    because we generate the signals. Validates end-to-end delineation +
    threshold logic (an optimistic bound; clearly synthetic).
  * ``build_ptbxl_dataset`` — loads real PTB-XL records + SCP labels from a
    local checkout (network to PhysioNet is not required at runtime; see
    README). Runs only if the dataset path exists.

Real clinical benchmarking (PTB-XL/MIT-BIH/LUDB) is the intended next step;
this harness is the scaffolding for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer6_interpret import StubProvider
from cardiolens.pipeline import run_pipeline
from cardiolens.validation.metrics import (
    IntervalError,
    macro_recall,
    score_findings,
)

# Finding labels scored by the harness.
SCORED_LABELS = [
    "sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia",
    "first_degree_av_block", "st_elevation", "peaked_t_waves",
]


@dataclass
class LabeledCase:
    signal: object
    gold_findings: set[str]
    gold_pr_ms: float | None = None
    gold_qt_ms: float | None = None
    name: str = ""


@dataclass
class ValidationReport:
    n_cases: int
    per_class: dict = field(default_factory=dict)
    macro_sensitivity: float | None = None
    pr_mae_ms: float | None = None
    qt_mae_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "macro_sensitivity": self.macro_sensitivity,
            "pr_mae_ms": self.pr_mae_ms,
            "qt_mae_ms": self.qt_mae_ms,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
        }


def build_synthetic_dataset() -> list[LabeledCase]:
    """A labelled synthetic dataset over reliably-stimulated patterns (TZ §15)."""
    cases: list[LabeledCase] = []

    # Sinus rhythm at several normal rates (explicit QT for interval MAE).
    for i, hr in enumerate((62, 70, 78, 88)):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=hr, pr_ms=150, qt_ms=380, seed=10 + i),
            {"sinus_rhythm"}, gold_pr_ms=150, gold_qt_ms=380, name=f"sinus_{hr}"))

    # Bradycardia.
    for i, hr in enumerate((40, 46, 52)):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=hr, pr_ms=160, seed=20 + i),
            {"sinus_bradycardia"}, gold_pr_ms=160, name=f"brady_{hr}"))

    # Tachycardia.
    for i, hr in enumerate((110, 125, 140)):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=hr, pr_ms=150, seed=30 + i),
            {"sinus_tachycardia"}, gold_pr_ms=150, name=f"tachy_{hr}"))

    # First-degree AV block.
    for i, pr in enumerate((240, 280, 300)):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=70, pr_ms=pr, seed=40 + i),
            {"sinus_rhythm", "first_degree_av_block"}, gold_pr_ms=pr, name=f"avb_{pr}"))

    # Inferior STEMI.
    for i in range(3):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=68 + i * 4, pr_ms=150,
                                   st_elevation_uv=250, st_leads=["II", "III", "aVF"],
                                   seed=50 + i),
            {"st_elevation"}, gold_pr_ms=150, name=f"stemi_{i}"))

    # Hyperkalemia (peaked T).
    for i in range(3):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=72, pr_ms=150, t_amp_uv=760,
                                   potassium=6.5, seed=60 + i),
            {"sinus_rhythm", "peaked_t_waves"}, gold_pr_ms=150, name=f"hyperk_{i}"))

    return cases


def run_validation(cases: list[LabeledCase], *, labels: list[str] | None = None) -> ValidationReport:
    labels = labels or SCORED_LABELS
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    pr_err = IntervalError("PR")
    qt_err = IntervalError("QT")

    for case in cases:
        result = run_pipeline(case.signal, mode="A", provider=StubProvider())
        preds.append(result.findings.codes())
        gold.append(case.gold_findings)
        pr_err.add(result.measurements.intervals.pr_ms, case.gold_pr_ms)
        qt_err.add(result.measurements.intervals.qt_ms, case.gold_qt_ms)

    per_class = score_findings(preds, gold, labels)
    return ValidationReport(
        n_cases=len(cases),
        per_class=per_class,
        macro_sensitivity=macro_recall(per_class),
        pr_mae_ms=pr_err.mae,
        qt_mae_ms=qt_err.mae,
    )


def build_ptbxl_dataset(root: str | Path, limit: int = 50) -> list[LabeledCase]:  # pragma: no cover
    """Load real PTB-XL records + SCP labels from a local checkout (TZ §15).

    Expects ``ptbxl_database.csv`` and ``scp_statements.csv`` at ``root`` and the
    WFDB records under ``records500/``. Maps a subset of SCP codes to our finding
    codes. Runs only when the dataset is present locally.
    """
    import ast

    import pandas as pd

    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    # Minimal SCP -> our-code mapping (extend as coverage grows).
    scp_map = {
        "SR": "sinus_rhythm", "SBRAD": "sinus_bradycardia", "STACH": "sinus_tachycardia",
        "1AVB": "first_degree_av_block", "AMI": "st_elevation", "IMI": "st_elevation",
        "CRBBB": "rbbb", "CLBBB": "lbbb", "AFIB": "atrial_fibrillation", "LVH": "lvh",
    }

    from cardiolens.layer1_ingestion import load_wfdb

    cases: list[LabeledCase] = []
    for ecg_id, row in db.head(limit).iterrows():
        rec = root / str(row["filename_hr"])
        if not rec.with_suffix(".hea").exists():
            continue
        gold = {scp_map[k] for k in row["scp_codes"] if k in scp_map}
        sig = load_wfdb(rec)
        sig.patient.age_years = float(row.get("age")) if row.get("age") == row.get("age") else None
        sig.patient.sex = "male" if row.get("sex") == 0 else "female"
        cases.append(LabeledCase(sig, gold, name=f"ptbxl_{ecg_id}"))
    return cases
