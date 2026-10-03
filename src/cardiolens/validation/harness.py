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

import re
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
from cardiolens.validation.risk import score_risk_tiers

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
    gold_qrs_ms: float | None = None
    name: str = ""
    meta: dict = field(default_factory=dict)   # dataset-specific extras (PTBDB: days, territory)


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
    qrs_mae_ms: float | None = None
    details: dict = field(default_factory=dict)   # name -> {"pred": codes, "st_leads": [...]}
    risk: dict = field(default_factory=dict)      # tier -> TierMetrics (error analysis by consequence)

    def to_dict(self) -> dict:
        return {
            "n_cases": self.n_cases,
            "macro_sensitivity": self.macro_sensitivity,
            "pr_mae_ms": self.pr_mae_ms,
            "qt_mae_ms": self.qt_mae_ms,
            "qrs_mae_ms": self.qrs_mae_ms,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "families": {k: v.to_dict() for k, v in self.families.items()},
            "risk": {k: v.to_dict() for k, v in self.risk.items()},
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
            generate_synthetic_ecg(heart_rate_bpm=72, pr_ms=150, t_amp_uv=1200,
                                   potassium=6.5, seed=60 + i),
            {"sinus_rhythm", "peaked_t_waves"}, gold_pr_ms=150, name=f"hyperk_{i}"))

    return cases


def run_validation(cases: list[LabeledCase], *, labels: list[str] | None = None) -> ValidationReport:
    labels = labels or SCORED_LABELS
    preds: list[set[str]] = []
    gold: list[set[str]] = []
    pr_err = IntervalError("PR")
    qt_err = IntervalError("QT")
    qrs_err = IntervalError("QRS")

    details: dict = {}
    for case in cases:
        result = run_pipeline(case.signal, mode="A", provider=StubProvider())
        codes = result.findings.codes()
        preds.append(codes)
        gold.append(case.gold_findings)
        pr_err.add(result.measurements.intervals.pr_ms, case.gold_pr_ms)
        qt_err.add(result.measurements.intervals.qt_ms, case.gold_qt_ms)
        qrs_err.add(result.measurements.intervals.qrs_ms, case.gold_qrs_ms)
        st = result.findings.by_code("st_elevation") if "st_elevation" in codes else None
        st_leads = next((e.value.split(",") for e in st.evidence if e.feature == "leads"), []) if st else []
        details[case.name] = {"pred": sorted(codes), "st_leads": st_leads}

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
        qrs_mae_ms=qrs_err.mae,
        details=details,
        risk=score_risk_tiers(preds, gold, labels=labels),
    )


# PTB-XL SCP code (and the fetch script's AXIS_* pseudo-labels) -> finding code.
PTBXL_SCP_MAP = {
    "SR": "sinus_rhythm", "SARRH": "sinus_rhythm",   # sinus arrhythmia is a sinus rhythm
    "SBRAD": "sinus_bradycardia", "STACH": "sinus_tachycardia",
    "AFIB": "atrial_fibrillation", "AFLT": "atrial_flutter",
    "1AVB": "first_degree_av_block", "LPR": "first_degree_av_block",
    "3AVB": "complete_av_block",
    "CRBBB": "rbbb", "CLBBB": "lbbb", "WPW": "wpw_preexcitation",
    "LVH": "lvh", "VCLVH": "lvh", "LNGQT": "long_qt", "LVOLT": "low_qrs_voltage",
    "STE_": "st_elevation", "STD_": "st_depression",
    # Subendocardial injury = ST depression of an ischaemic territory (D42).
    "INJAS": "st_depression", "INJAL": "st_depression", "INJIN": "st_depression",
    "INJLA": "st_depression", "INJIL": "st_depression",
    "LAO/LAE": "left_atrial_enlargement", "RAO/RAE": "right_atrial_enlargement",
    # PTB-XL heart_axis categories (D49): LAD is the *mild* leftward axis (12SL
    # axis median -16, -30..0), ALAD the abnormal one (median -48) — only ALAD is
    # left axis deviation by the -30 criterion; RAD (median 96) and ARAD (> 120,
    # north-west) are both right axis deviation.
    "AXIS_ALAD": "left_axis_deviation", "AXIS_RAD": "right_axis_deviation",
    "AXIS_ARAD": "right_axis_deviation",
}

_AXIS_CODES = {"ALAD": "AXIS_ALAD", "RAD": "AXIS_RAD", "ARAD": "AXIS_ARAD"}


def ptbxl_axis_labels(heart_axis: str | None) -> set[str]:
    """Pseudo-label(s) for a PTB-XL ``heart_axis`` category (see PTBXL_SCP_MAP)."""
    code = _AXIS_CODES.get((heart_axis or "").strip())
    return {code} if code else set()


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
            labels = {k for k in rec.get("labels", []) if not k.startswith("AXIS_")}
            labels |= ptbxl_axis_labels(rec.get("heart_axis"))
            gold = {PTBXL_SCP_MAP[k] for k in labels if k in PTBXL_SCP_MAP}
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


# PTB Diagnostic ECG Database (PhysioNet ptbdb): acute-MI localisation -> ECG
# territories the STEMI alert can name (posterior has no standard-lead territory).
PTBDB_TERRITORY = {
    "inferior": {"inferior"}, "infero-lateral": {"inferior", "lateral"},
    "infero-latera": {"inferior", "lateral"}, "infero-posterior": {"inferior"},
    "infero-postero-lateral": {"inferior", "lateral"}, "infero-poster-lateral": {"inferior", "lateral"},
    "anterior": {"anterior"}, "antero-septal": {"anterior"}, "antero-lateral": {"anterior", "lateral"},
    "antero-septo-lateral": {"anterior", "lateral"}, "lateral": {"lateral"},
    "postero-lateral": {"lateral"}, "posterior": set(),
}
LEAD_TERRITORY = {"II": "inferior", "III": "inferior", "aVF": "inferior",
                  "V1": "anterior", "V2": "anterior", "V3": "anterior", "V4": "anterior",
                  "I": "lateral", "aVL": "lateral", "V5": "lateral", "V6": "lateral"}


def territories_of_leads(leads) -> set[str]:
    """ECG territories named by a set of leads (for alert/gold agreement)."""
    return {LEAD_TERRITORY[lead] for lead in leads if lead in LEAD_TERRITORY}


def build_ptbdb_dataset(root: str | Path, limit: int = 5000, max_days: int = 3,
                        seconds: float = 10.0) -> list[LabeledCase]:  # pragma: no cover
    """PTBDB as a STEMI benchmark (TZ §16, open item 1).

    Gold ``st_elevation`` = admission for myocardial infarction with the ECG recorded
    at most ``max_days`` after the acute infarction (a proxy: the ST elevation of a
    STEMI is expected in the first days and fades afterwards). Later MI ECGs, healthy
    controls and every other diagnosis are negatives. ``meta`` carries ``days``,
    ``reason`` and the gold ``territory`` set for the day-bin / territory report.
    Only the first ``seconds`` of the 12 standard leads are read.
    """
    import json

    from cardiolens.layer1_ingestion import load_wfdb

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases: list[LabeledCase] = []
    for rec in manifest["records"][:limit]:
        path = root / rec["record"]
        if not path.with_suffix(".hea").exists() or not path.with_suffix(".dat").exists():
            continue
        try:
            sig = load_wfdb(path, sampto=int(seconds * 1000), channels=list(range(12)))
        except Exception:  # noqa: S112 - skip unreadable records
            continue
        sig.patient.age_years = rec.get("age")
        sig.patient.sex = rec.get("sex")
        days = rec.get("days_since_infarction")
        is_mi = rec.get("reason", "").startswith("Myocardial infarction")
        gold: set[str] = set()
        if is_mi and days is not None and 0 <= days <= max_days:
            gold.add("st_elevation")
        territory = PTBDB_TERRITORY.get(rec.get("acute_localization", ""), set()) if is_mi else set()
        cases.append(LabeledCase(sig, gold, name=f"ptbdb_{rec['record'].replace('/', '_')}",
                                 meta={"days": days, "reason": rec.get("reason", ""),
                                       "territory": territory, "mi": is_mi}))
    return cases


PTBDB_DAY_BINS = [(0, 1), (2, 3), (4, 7), (8, 14), (15, 10**6)]


def ptbdb_report_lines(cases: list[LabeledCase], report: ValidationReport) -> list[str]:
    """STEMI-alert sensitivity by days since infarction, territory agreement, and
    false positives by admission reason (PTBDB, open item 1)."""
    lines: list[str] = []
    det = report.details
    mi = [c for c in cases if c.meta.get("mi") and c.meta.get("days") is not None]
    for lo, hi in PTBDB_DAY_BINS:
        sub = [c for c in mi if lo <= c.meta["days"] <= hi]
        if not sub:
            continue
        alert = sum("st_elevation" in det[c.name]["pred"] for c in sub)
        anyst = sum(bool({"st_elevation", "st_elevation_nonspecific", "st_elevation_any"}
                         & set(det[c.name]["pred"])) for c in sub)
        agree = sum(1 for c in sub if "st_elevation" in det[c.name]["pred"] and c.meta["territory"]
                    and territories_of_leads(det[c.name]["st_leads"]) & c.meta["territory"])
        label = f"{lo}-{hi} d" if hi < 10**6 else f">= {lo} d"
        lines.append(f"MI {label:9s} n={len(sub):3d} alert {alert/len(sub):.2f} ({alert})  "
                     f"any-ST {anyst/len(sub):.2f}  territory agreement {agree}/{alert}")
    by_reason: dict[str, list[int]] = {}
    for c in cases:
        if c.meta.get("mi"):
            continue
        r = c.meta.get("reason") or "n/a"
        by_reason.setdefault(r, [0, 0])
        by_reason[r][0] += 1
        by_reason[r][1] += "st_elevation" in det[c.name]["pred"]
    for r, (n, fp) in sorted(by_reason.items(), key=lambda kv: -kv[1][0]):
        lines.append(f"non-MI {r[:32]:32s} n={n:3d} alert false positives {fp} (spec {1 - fp / n:.3f})")
    return lines


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
        s = re.sub(r"^#\s+", "#", line.strip().lower())   # "# Age:" in Challenge-2020 headers
        if s.startswith("#age:"):
            try:
                age = float(s.split(":", 1)[1].strip())
            except ValueError:
                age = None
            # ecg-arrhythmia writes 0 for an unknown age: the 247 "age 0" records
            # measure like adults (HR median 78, QRS 90 ms), not like infants (D44).
            if age == 0:
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
        s = re.sub(r"^#\s+", "#", line.strip())   # Challenge-2020 headers write "# Dx:"
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


# --------------------------------------------------------------------------- #
# LUDB — cardiologist-marked wave boundaries on every lead (TZ §16 interval MAE)
# --------------------------------------------------------------------------- #
LUDB_LEAD_FILES = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
_LUDB_RHYTHM = {
    "sinus rhythm": "sinus_rhythm", "sinus bradycardia": "sinus_bradycardia",
    "sinus tachycardia": "sinus_tachycardia", "sinus arrhythmia": "sinus_rhythm",
    "atrial fibrillation": "atrial_fibrillation", "atrial flutter": "atrial_flutter",
}
_LUDB_DIAG = {
    "left axis deviation": "left_axis_deviation", "right axis deviation": "right_axis_deviation",
    "left ventricular hypertrophy": "lvh", "complete right bundle branch block": "rbbb",
    "complete left bundle branch block": "lbbb", "i degree av block": "first_degree_av_block",
    "first degree av block": "first_degree_av_block", "wpw": "wpw_preexcitation",
    "stemi": "st_elevation",   # "STEMI: anterior wall" etc. — the only ST-elevation label in LUDB
}


def ludb_intervals_from_annotations(root: Path, name: str, fs: float) -> dict[str, float | None]:
    """Median PR / QRS / QT (ms) over the beats of every annotated lead, then over leads.

    Per lead the annotation stream is ``( p ) ( N ) ( t )`` triplets; a beat's PR
    is its P onset to its QRS onset, QRS the onset to offset, QT the QRS onset to
    the T offset (the T belonging to that QRS). Beats without a P (AF) give no PR.
    """
    import numpy as np
    import wfdb

    per_lead: dict[str, list[float]] = {"pr": [], "qrs": [], "qt": []}
    for lead in LUDB_LEAD_FILES:
        if not (root / f"{name}.{lead}").exists():
            continue
        ann = wfdb.rdann(str(root / name), lead)
        sym, smp = list(ann.symbol), [int(x) for x in ann.sample]
        waves = []   # (kind, on, off)
        i = 0
        while i + 2 < len(sym):
            if sym[i] == "(" and sym[i + 2] == ")" and sym[i + 1] in ("p", "N", "t"):
                waves.append((sym[i + 1], smp[i], smp[i + 2]))
                i += 3
            else:
                i += 1
        pr, qrs, qt = [], [], []
        for k, (kind, on, off) in enumerate(waves):
            if kind != "N":
                continue
            qrs.append((off - on) / fs * 1000.0)
            if k > 0 and waves[k - 1][0] == "p" and on - waves[k - 1][1] < 0.4 * fs:
                pr.append((on - waves[k - 1][1]) / fs * 1000.0)
            if k + 1 < len(waves) and waves[k + 1][0] == "t" and waves[k + 1][2] - on < 0.7 * fs:
                qt.append((waves[k + 1][2] - on) / fs * 1000.0)
        for key, vals in (("pr", pr), ("qrs", qrs), ("qt", qt)):
            if vals:
                per_lead[key].append(float(np.median(vals)))
    return {key: (float(np.median(vals)) if vals else None) for key, vals in per_lead.items()}


def build_ludb_dataset(root: str | Path, limit: int = 5000) -> list[LabeledCase]:  # pragma: no cover
    """LUDB as the interval-MAE benchmark plus its header diagnoses as gold findings."""
    import json

    import numpy as np
    import wfdb

    from cardiolens.contracts.signal import Patient, SignalRepresentation, canonical_lead_name

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    names = sorted((p.stem for p in root.glob("*.hea")), key=lambda x: int(x) if x.isdigit() else x)[:limit]
    cases = []
    for name in names:
        rec = wfdb.rdrecord(str(root / name))
        fs = float(rec.fs)
        lead_names = [canonical_lead_name(ln) for ln in rec.sig_name]
        signal = (rec.p_signal.T * 1000.0).astype(np.float32)   # mV -> µV
        meta = manifest.get(name) or {}
        gold: set[str] = set()
        rhythm = (meta.get("rhythm") or "").lower()
        for key, code in _LUDB_RHYTHM.items():
            if key in rhythm:
                gold.add(code)
        for diag in [meta.get("axis") or ""] + list(meta.get("diagnoses") or []):
            d = diag.lower()
            for key, code in _LUDB_DIAG.items():
                # "complete right bundle branch block" is a substring of the
                # incomplete form: match whole phrases only.
                if re.search(r"(?<![a-z])" + re.escape(key), d):
                    gold.add(code)
        ivs = ludb_intervals_from_annotations(root, name, fs)
        sig = SignalRepresentation(
            signal=signal, sampling_rate_hz=fs, lead_names=lead_names,
            duration_s=float(signal.shape[1] / fs), source="wfdb",
            patient=Patient(age_years=meta.get("age"), sex=meta.get("sex")),
        )
        cases.append(LabeledCase(signal=sig, gold_findings=gold, gold_pr_ms=ivs["pr"], gold_qt_ms=ivs["qt"],
                                 gold_qrs_ms=ivs["qrs"], name=f"ludb_{name}", meta=meta))
    return cases


# MIMIC-IV-ECG cart statements (upper-cased) -> finding codes. An algorithm's
# reading, not a physician's — but an explicit ECG-level statement, which is what
# the STEMI alert and the hyperkalaemia T pattern lack everywhere else (D62).
_MIMIC_STATEMENTS: list[tuple[str, str, str | None]] = [
    # (regex, code, exclusion regex applied to the same statement)
    (r"\*\*\* CONSIDER ACUTE ST ELEVATION MI \*\*\*|ST ELEVATION, CONSIDER ACUTE INFARCT|INFARCT, ACUTE\b", "st_elevation", None),
    (r"TALL T WAVES", "peaked_t_waves", None),
    (r"PROLONGED QT", "long_qt", None),
    (r"LEFT BUNDLE BRANCH BLOCK", "lbbb", r"INCOMPLETE"),
    (r"RIGHT BUNDLE BRANCH BLOCK", "rbbb", r"INCOMPLETE"),
    (r"ATRIAL FIBRILLATION", "atrial_fibrillation", None),
    (r"ATRIAL FLUTTER", "atrial_flutter", None),
    (r"SINUS BRADYCARDIA", "sinus_bradycardia", None),
    (r"SINUS TACHYCARDIA", "sinus_tachycardia", None),
    # as in the SNOMED map: a sinus brady-/tachycardia statement is that code alone
    (r"\bSINUS RHYTHM\b|\bSINUS ARRHYTHMIA\b", "sinus_rhythm", None),
    (r"1ST DEGREE A-?V BLOCK", "first_degree_av_block", r"BORDERLINE"),
    (r"COMPLETE HEART BLOCK|3RD DEGREE A-?V BLOCK|COMPLETE A-?V BLOCK", "complete_av_block", None),
    (r"LEFT VENTRICULAR HYPERTROPHY", "lvh", r"POSSIBLE|CONSIDER|CANNOT"),
    (r"LEFT AXIS DEVIATION", "left_axis_deviation", r"BORDERLINE"),
    (r"RIGHT AXIS DEVIATION", "right_axis_deviation", r"BORDERLINE"),
    (r"LOW QRS VOLTAGE", "low_qrs_voltage", None),
    (r"WOLFF|WPW|PRE-?EXCITATION", "wpw_preexcitation", None),
    (r"ST DEPRESSION", "st_depression", r"JUNCTIONAL|NONSPECIFIC"),
]


def mimic_statements_to_codes(statements) -> set[str]:
    """Finding codes named by a record's cart statements (see ``_MIMIC_STATEMENTS``)."""
    codes: set[str] = set()
    for s in statements:
        up = str(s).upper()
        for pat, code, excl in _MIMIC_STATEMENTS:
            if re.search(pat, up) and not (excl and re.search(excl, up)):
                codes.add(code)
    return codes


def build_mimic_dataset(root: str | Path, limit: int = 5000) -> list[LabeledCase]:  # pragma: no cover
    """MIMIC-IV-ECG subset fetched by ``fetch_mimic``: gold = the cart's statements.

    No age or sex is available in the open files (adult norms apply; MIMIC holds
    adults only). ``meta`` carries the selection classes and the raw statements."""
    import json

    from cardiolens.layer1_ingestion import load_wfdb

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cases: list[LabeledCase] = []
    for rec in manifest["records"][:limit]:
        path = root / rec["record"]
        if not path.with_suffix(".hea").exists() or not path.with_suffix(".dat").exists():
            continue
        try:
            sig = load_wfdb(path)
        except Exception:  # noqa: S112 - skip unreadable records
            continue
        cases.append(LabeledCase(sig, mimic_statements_to_codes(rec.get("statements", [])),
                                 name=f"mimic_{rec['study_id']}",
                                 meta={"classes": rec.get("classes", []), "statements": rec.get("statements", [])}))
    return cases
