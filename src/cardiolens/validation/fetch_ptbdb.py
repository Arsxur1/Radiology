"""Fetch the PTB Diagnostic ECG Database (PhysioNet ``ptbdb``) for STEMI validation.

    python -m cardiolens.validation.fetch_ptbdb --out data/ptbdb

549 records from 290 subjects, 15 leads (12 standard + Frank XYZ), 1 kHz, ~2 min each
(~1 MB per ``.dat``). Every header carries the admission diagnosis, the acute-infarction
localisation, the infarction date and the ECG date — the only reachable source of
ECGs recorded *days* after an acute myocardial infarction, i.e. the closest thing to
a STEMI gold standard (TZ §16, item 1 of the open list). Downloads every record with
the keep-alive client from ``fetch_physionet`` (resumable), writes ``manifest.json``
with the parsed metadata. Licensed data — never committed (``data/ptbdb/`` is
git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from cardiolens.validation.fetch_physionet import fetch_bytes

BASE_URL = "https://physionet.org/files/ptbdb/1.0.0/"

_FIELDS = {
    "reason": "Reason for admission",
    "acute_localization": "Acute infarction (localization)",
    "former_localization": "Former infarction (localization)",
    "infarction_date": "Infarction date (acute)",
    "ecg_date": "ECG date",
    "age": "age",
    "sex": "sex",
    "additional": "Additional diagnoses",
}


def parse_header_comments(text: str) -> dict:
    """Metadata dict from a ptbdb ``.hea`` (comment lines ``# key: value``)."""
    out: dict = {}
    for key, label in _FIELDS.items():
        m = re.search(rf"^# {re.escape(label)}: *(.*)$", text, flags=re.MULTILINE)
        out[key] = m.group(1).strip() if m else ""
    out["days_since_infarction"] = days_between(out["infarction_date"], out["ecg_date"])
    try:
        out["age"] = float(out["age"])
    except ValueError:
        out["age"] = None
    out["sex"] = {"male": "male", "female": "female"}.get(out["sex"].lower())
    return out


def days_between(infarction: str, ecg: str) -> int | None:
    """ECG date (dd/mm/yyyy) minus infarction date (dd-Mon-yy), in days."""
    try:
        d_ecg = datetime.strptime(ecg, "%d/%m/%Y")
        d_inf = datetime.strptime(infarction, "%d-%b-%y")
    except ValueError:
        return None
    return (d_ecg - d_inf).days


SECONDS = 10.0   # only the first 10 s are downloaded (format 16, 15 leads, 1 kHz)


def truncate_header(text: str, n_samples: int) -> str:
    """Header for the first ``n_samples`` (the .dat is cut to match; checksums stay,
    wfdb does not verify them on read)."""
    lines = text.splitlines()
    first = lines[0].split()
    first[3] = str(n_samples)
    lines[0] = " ".join(first)
    return "\n".join(lines) + "\n"


def _fetch_record(base: str, rec: str, out: Path) -> dict:
    dst = out / rec
    dst.parent.mkdir(parents=True, exist_ok=True)
    hea = dst.with_suffix(".hea")
    if not hea.exists():
        raw = fetch_bytes(base + rec + ".hea").decode("utf-8", errors="ignore")
        fields = raw.splitlines()[0].split()
        n_sig, fs = int(fields[1]), float(fields[2])
        n_samples = int(SECONDS * fs)
        dat = dst.with_suffix(".dat")
        dat.write_bytes(fetch_bytes(base + rec + ".dat", max_bytes=n_samples * n_sig * 2))
        hea.write_text(truncate_header(raw, n_samples), encoding="utf-8")
    meta = parse_header_comments(hea.read_text(encoding="utf-8", errors="ignore"))
    meta["record"] = rec
    meta["patient"] = rec.split("/")[0]
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch the PTB Diagnostic ECG Database")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N records only (0 = all)")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = fetch_bytes(BASE_URL + "RECORDS").decode().split()
    if args.limit:
        records = records[: args.limit]
    print(f"[ptbdb] {len(records)} records -> {out}", file=sys.stderr, flush=True)
    metas: list[dict] = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, meta in enumerate(pool.map(lambda r: _fetch_record(BASE_URL, r, out), records), 1):
            metas.append(meta)
            if i % 50 == 0:
                print(f"[ptbdb] {i}/{len(records)}", file=sys.stderr, flush=True)
    manifest = {"source": BASE_URL, "records": metas}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reasons = {}
    for m in metas:
        reasons[m["reason"]] = reasons.get(m["reason"], 0) + 1
    print(f"[ptbdb] done; reasons: {reasons}", file=sys.stderr)


if __name__ == "__main__":
    main()
