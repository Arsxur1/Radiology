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

# Finding labels scored by the harness. Covers the rhythm/conduction/axis/ischemia
# classes the pipeline emits (layer4_findings + safety) that also appear as gold
# labels in the real datasets (PTB-XL SCP, ecg-arrhythmia SNOMED). The synthetic
# dataset only stimulates a subset; unsupported classes simply report support=0.
SCORED_LABELS = [
    "sinus_rhythm", "sinus_bradycardia", "sinus_tachycardia",
    "atrial_fibrillation", "atrial_flutter", "first_degree_av_block",
    "complete_av_block", "rbbb", "lbbb",
    "left_axis_deviation", "right_axis_deviation", "low_qrs_voltage", "lvh",
    "wpw_preexcitation",
    "st_elevation", "st_depression", "long_qt", "peaked_t_waves",
]


@dataclass
class LabeledCase:
    signal: object
    gold_findings: set[str]
    gold_pr_ms: float | None = None
    gold_qt_ms: float | None = None
    name: str = ""


# Derived "family" scores: a prediction of any code in the first set against a gold
# label of any code in the second. AF and flutter with an irregular response are one
# clinical family (the rule set cannot always tell them apart); "any ST elevation"
# matches the broad dataset labels (ST tilt up / non-specific ST elevation) that the
# STEMI-grade alert deliberately does not.
FAMILIES: dict[str, tuple[set[str], set[str]]] = {
    "af_or_flutter": ({"atrial_fibrillation", "atrial_flutter"},
                      {"atrial_fibrillation", "atrial_flutter"}),
    "st_elevation_any": ({"st_elevation", "st_elevation_nonspecific", "st_elevation_any"},
                         {"st_elevation"}),
}


@dataclass
class ValidationReport:
    n_cases: int
    per_class: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
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
            "families": {k: v.to_dict() for k, v in self.families.items()},
        }


def filter_by_age(cases: list[LabeledCase], age_min: float | None = None,
                  age_max: float | None = None) -> list[LabeledCase]:
    """Keep cases whose patient age is known and inside [age_min, age_max]."""
    out = []
    for c in cases:
        age = getattr(c.signal.patient, "age_years", None)
        if age is None:
            continue
        if (age_min is None or age >= age_min) and (age_max is None or age <= age_max):
            out.append(c)
    return out


def build_synthetic_dataset() -> list[LabeledCase]:
    """A labelled synthetic dataset over reliably-stimulated patterns (TZ §15)."""
    cases: list[LabeledCase] = []

    # Sinus rhythm at several normal rates (rate-appropriate QT so QTc stays normal;
    # a fixed 380 ms at 88 bpm is a Bazett long-QT by definition).
    for i, hr in enumerate((62, 70, 78, 88)):
        qt = round(400.0 * (60.0 / hr) ** 0.5)
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=hr, pr_ms=150, qt_ms=qt, seed=10 + i),
            {"sinus_rhythm"}, gold_pr_ms=150, gold_qt_ms=qt, name=f"sinus_{hr}"))

    # Bradycardia.
    for i, hr in enumerate((40, 46, 52)):
        cases.append(LabeledCase(
            generate_synthetic_ecg(heart_rate_bpm=hr, pr_ms=160, seed=20 + i),
            {"sinus_bradycardia"}, gold_pr_ms=160, name=f"brady_{hr}"))

    # Tachycardia.
    for i, hr in enumerate((110, 120, 130)):
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
    families = {}
    for name, (pred_codes, gold_codes) in FAMILIES.items():
        fam_pred = [{name} if p & pred_codes else set() for p in preds]
        fam_gold = [{name} if g & gold_codes else set() for g in gold]
        families[name] = score_findings(fam_pred, fam_gold, [name])[name]
    return ValidationReport(
        n_cases=len(cases),
        per_class=per_class,
        families=families,
        macro_sensitivity=macro_recall(per_class),
        pr_mae_ms=pr_err.mae,
        qt_mae_ms=qt_err.mae,
    )


# PTB-XL SCP code (and the fetch script's AXIS_* pseudo-labels) -> finding code.
PTBXL_SCP_MAP = {
    "SR": "sinus_rhythm", "SBRAD": "sinus_bradycardia", "STACH": "sinus_tachycardia",
    "AFIB": "atrial_fibrillation", "AFLT": "atrial_flutter",
    "1AVB": "first_degree_av_block", "LPR": "first_degree_av_block",
    "3AVB": "complete_av_block",
    "CRBBB": "rbbb", "CLBBB": "lbbb", "WPW": "wpw_preexcitation",
    "LVH": "lvh", "VCLVH": "lvh", "LNGQT": "long_qt", "LVOLT": "low_qrs_voltage",
    "STE_": "st_elevation", "STD_": "st_depression",
    "LAO/LAE": "left_atrial_enlargement", "RAO/RAE": "right_atrial_enlargement",
    "AXIS_LAD": "left_axis_deviation", "AXIS_RAD": "right_axis_deviation",
}


def build_ptbxl_dataset(root: str | Path, limit: int = 50) -> list[LabeledCase]:  # pragma: no cover
    """Load PTB-XL records from a local checkout (TZ §15).

    With a ``manifest.json`` from ``fetch_ptbxl`` the selection, labels (SCP codes
    with likelihood >= 50), axis and patient age/sex come from the manifest.
    Without one, falls back to ``ptbxl_database.csv`` + whatever ``records500/``
    files exist. Runs only when the data is present locally.
    """
    import json

    from cardiolens.layer1_ingestion import load_wfdb

    root = Path(root)
    manifest = root / "manifest.json"
    cases: list[LabeledCase] = []
    if manifest.exists():
        for rec in json.loads(manifest.read_text(encoding="utf-8"))["records"][:limit]:
            path = root / rec["filename_hr"]
            if not path.with_suffix(".hea").exists() or not path.with_suffix(".dat").exists():
                continue
            gold = {PTBXL_SCP_MAP[k] for k in rec.get("labels", []) if k in PTBXL_SCP_MAP}
            try:
                sig = load_wfdb(path)
            except Exception:  # noqa: S112 - skip unreadable records
                continue
            sig.patient.age_years = rec.get("age")
            sex = rec.get("sex")
            sig.patient.sex = ("female" if str(sex) in ("1", "1.0", "female")
                               else "male" if str(sex) in ("0", "0.0", "male") else None)
            cases.append(LabeledCase(sig, gold, name=f"ptbxl_{rec['ecg_id']}"))
        return cases

    import ast

    import pandas as pd

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    for ecg_id, row in db.head(limit).iterrows():
        rec = root / str(row["filename_hr"])
        if not rec.with_suffix(".hea").exists():
            continue
        gold = {PTBXL_SCP_MAP[k] for k, v in row["scp_codes"].items()
                if k in PTBXL_SCP_MAP and (float(v) >= 50 or float(v) == 0)}
        sig = load_wfdb(rec)
        sig.patient.age_years = float(row.get("age")) if row.get("age") == row.get("age") else None
        sig.patient.sex = "male" if row.get("sex") == 0 else "female"
        cases.append(LabeledCase(sig, gold, name=f"ptbxl_{ecg_id}"))
    return cases


# SNOMED-CT concept id -> CardioLens finding code, for the PhysioNet
# ecg-arrhythmia (Chapman-Shaoxing / Ningbo) 12-lead database, following the
# dataset's ConditionNames_SNOMED-CT.csv. Extend as finding coverage grows;
# unmapped codes are ignored during scoring. Label-noise caveat: the dataset maps
# the fascicular blocks (LFBBB/LBBBB, narrow QRS) to the same code as complete LBBB.
SNOMED_TO_CODE = {
    "426783006": "sinus_rhythm",
    "426177001": "sinus_bradycardia",
    "427084000": "sinus_tachycardia",
    "164889003": "atrial_fibrillation",
    "164890007": "atrial_flutter",
    "270492004": "first_degree_av_block",
    "164947007": "first_degree_av_block",   # "PR interval extension"
    "27885002": "complete_av_block",
    "733534002": "lbbb", "164909002": "lbbb",
    "713427006": "rbbb", "59118001": "rbbb",
    "164931005": "st_elevation",           # "ST tilt up"
    "164930006": "st_elevation",           # "ST extension"
    "429622005": "st_depression",
    "39732003": "left_axis_deviation",
    "47665007": "right_axis_deviation",
    "164873001": "lvh",
    "251146004": "low_qrs_voltage",        # all leads
    "251147008": "low_qrs_voltage",        # limb leads (what the detector measures)
    "74390002": "wpw_preexcitation",
    "195060002": "wpw_preexcitation",      # "ventricular preexcitation"
    "111975006": "long_qt",
}


def snomed_dx_to_codes(dx_codes) -> set[str]:
    """Map a list of SNOMED-CT ids (str/int) to CardioLens finding codes."""
    return {SNOMED_TO_CODE[str(c).strip()] for c in dx_codes
            if str(c).strip() in SNOMED_TO_CODE}


def parse_patient_from_header(header_text: str) -> tuple[float | None, str | None]:
    """(age_years, sex) from the ``#Age:`` / ``#Sex:`` header comments (None if absent)."""
    age: float | None = None
    sex: str | None = None
    for line in header_text.splitlines():
        s = line.strip().lower()
        if s.startswith("#age:"):
            try:
                age = float(s.split(":", 1)[1].strip())
            except ValueError:
                age = None
        elif s.startswith("#sex:"):
            v = s.split(":", 1)[1].strip()
            sex = "female" if v.startswith("f") else "male" if v.startswith("m") else None
    return age, sex


def parse_dx_from_header(header_text: str) -> list[str]:
    """Extract SNOMED-CT diagnosis ids from a WFDB header's ``#Dx:`` comment.

    ecg-arrhythmia stores labels as a header comment, e.g. ``#Dx: 426783006,39732003``.
    """
    for line in header_text.splitlines():
        s = line.strip()
        if s.lower().startswith("#dx:"):
            payload = s.split(":", 1)[1]
            return [c.strip() for c in payload.replace(";", ",").split(",") if c.strip()]
    return []


def build_ecg_arrhythmia_dataset(root, limit: int = 50) -> list[LabeledCase]:  # pragma: no cover
    """Load the PhysioNet ecg-arrhythmia (Chapman/Ningbo) dataset from a local checkout.

    Expects WFDB records (``.hea`` + ``.mat``) under ``root`` (e.g. the
    ``WFDBRecords/`` tree). Diagnostic labels come from each header's ``#Dx:``
    SNOMED-CT comment, mapped via ``SNOMED_TO_CODE``. Runs only when the dataset
    is present locally — PhysioNet is not reachable from this environment
    (egress policy); see docs/CLINICAL_REFERENCES.md for how to obtain it.
    """
    from cardiolens.layer1_ingestion import load_wfdb

    root = Path(root)
    # Only records with a signal file, outside hidden dirs (the fetch script keeps a
    # header-only cache under ``.headers/``).
    headers = [h for h in sorted(root.rglob("*.hea"))
               if h.with_suffix(".mat").exists()
               and not any(part.startswith(".") for part in h.relative_to(root).parts)][:limit]
    cases: list[LabeledCase] = []
    for hea in headers:
        text = hea.read_text(encoding="utf-8")
        gold = snomed_dx_to_codes(parse_dx_from_header(text))
        try:
            sig = load_wfdb(hea.with_suffix(""))
        except Exception:  # noqa: S112 - skip unreadable records, keep validating the rest
            continue
        # Age/sex drive the norm tables (pediatric bands, sex-specific QTc / V2-V3).
        age, sex = parse_patient_from_header(text)
        if age is not None:
            sig.patient.age_years = age
        if sex is not None:
            sig.patient.sex = sex
        cases.append(LabeledCase(sig, gold, name=f"ecgarr_{hea.stem}"))
    return cases
