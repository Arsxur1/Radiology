"""Fetch a statement-selected subset of MIMIC-IV-ECG (PhysioNet ``mimic-iv-ecg`` 1.0).

    python -m cardiolens.validation.fetch_mimic --out data/mimic-iv-ecg --per-class 300

800 035 ten-second 12-lead ECGs (500 Hz) from Beth Israel Deaconess, open access
(ODbL). ``machine_measurements.csv`` carries the cart's interpretation statements
(``report_0`` … ``report_17``). They are an algorithm's reading, not a physician's,
but they are an **explicit ECG-level label** — "*** CONSIDER ACUTE ST ELEVATION
MI ***", "TALL T WAVES - CONSIDER ACUTE ISCHEMIA OR HYPERKALEMIA", "PROLONGED QT"
— which no other reachable database offers for STEMI or the hyperkalaemia T
pattern. Records are selected per statement class with a fixed seed, downloaded
(``.hea`` + ``.dat``, ~120 KB each) under the dataset's own path layout
``files/pXXXX/pSUBJECT/sSTUDY/STUDY`` and listed in ``manifest.json`` with their
statements. Licensed data — never committed (``data/mimic-iv-ecg/`` is git-ignored).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cardiolens.validation.fetch_physionet import fetch_bytes

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"

# Statement classes → regular expressions over the joined, upper-cased report lines.
CLASSES: dict[str, str] = {
    "stemi": r"\*\*\* CONSIDER ACUTE ST ELEVATION MI \*\*\*|ST ELEVATION, CONSIDER ACUTE INFARCT|INFARCT, ACUTE\b",
    "normal": r"\bNORMAL ECG\b",
    "tall_t": r"TALL T WAVES",
    "long_qt": r"PROLONGED QT",
    "lbbb": r"LEFT BUNDLE BRANCH BLOCK",
}


def record_path(subject_id: int, study_id: int) -> str:
    return f"files/p{str(subject_id)[:4]}/p{subject_id}/s{study_id}/{study_id}"


def load_statements(csv_path: Path):  # pragma: no cover - pandas on a 180 MB file
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=lambda c: c.startswith("report_") or c in ("subject_id", "study_id", "ecg_time"),
                     low_memory=False)
    rep = [c for c in df.columns if c.startswith("report_")]
    text = df[rep].fillna("").astype(str).agg(" | ".join, axis=1).str.upper()
    return df, text


def select(df, text, per_class: int, seed: int = 0) -> list[dict]:  # pragma: no cover
    rng = np.random.default_rng(seed)
    chosen: dict[int, dict] = {}
    for cls, pat in CLASSES.items():
        mask = text.str.contains(pat, regex=True)
        if cls == "normal":                       # a normal must carry no other class statement
            for other, opat in CLASSES.items():
                if other != "normal":
                    mask &= ~text.str.contains(opat, regex=True)
        idx = np.flatnonzero(mask.to_numpy())
        pick = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        for i in pick:
            row = df.iloc[int(i)]
            sid = int(row.study_id)
            if sid in chosen:
                chosen[sid]["classes"].append(cls)
                continue
            statements = [s.strip() for s in text.iloc[int(i)].split(" | ") if s.strip()]
            chosen[sid] = {"subject_id": int(row.subject_id), "study_id": sid, "ecg_time": str(row.ecg_time),
                           "record": record_path(int(row.subject_id), sid), "classes": [cls], "statements": statements}
    return list(chosen.values())


def fetch(out: Path, per_class: int, seed: int = 0, limit: int | None = None) -> int:  # pragma: no cover
    csv_path = out / "machine_measurements.csv"
    if not csv_path.exists():
        csv_path.write_bytes(fetch_bytes(BASE_URL + "machine_measurements.csv"))
    df, text = load_statements(csv_path)
    records = select(df, text, per_class, seed)
    if limit:
        records = records[:limit]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": BASE_URL, "records": []}
    have = {r["study_id"] for r in manifest["records"]}
    for n, rec in enumerate(records, 1):
        dst = out / rec["record"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".hea", ".dat"):
            p = dst.with_suffix(ext)
            if p.exists() and p.stat().st_size > 0:
                continue
            p.write_bytes(fetch_bytes(BASE_URL + rec["record"] + ext))
        if rec["study_id"] not in have:
            manifest["records"].append(rec)
            have.add(rec["study_id"])
        if n % 20 == 0 or n == len(records):
            manifest_path.write_text(json.dumps(manifest, indent=1))
            print(f"{n}/{len(records)}", file=sys.stderr, flush=True)
    return len(records)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/mimic-iv-ecg")
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    n = fetch(Path(args.out), args.per_class, args.seed, args.limit)
    print(f"fetched {n} records into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
