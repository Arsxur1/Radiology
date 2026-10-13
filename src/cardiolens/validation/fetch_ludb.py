"""Fetch the Lobachevsky University Electrocardiography Database (PhysioNet ``ludb``).

    python -m cardiolens.validation.fetch_ludb --out data/ludb

200 ten-second 12-lead records at 500 Hz with cardiologist-marked P, QRS and T
onsets/offsets on every lead (one WFDB annotation file per lead: ``N.i``,
``N.ii`` … ``N.v6``) and header comments with age, sex, rhythm, axis and
diagnoses. The only reachable source of annotated wave boundaries — the interval
MAE benchmark of TZ §16. Downloads everything with the keep-alive client
(resumable), writes ``manifest.json`` with the parsed header comments. Licensed
data — never committed (``data/ludb/`` is git-ignored).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from cardiolens.validation.fetch_physionet import fetch_bytes

BASE_URL = "https://physionet.org/files/ludb/1.0.1/"
LEAD_FILES = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")


def parse_header_comments(text: str) -> dict:
    """``#<age>: 51`` / ``#<sex>: F`` / ``#Rhythm: …`` / ``#Electric axis …`` lines."""
    out: dict = {"age": None, "sex": None, "rhythm": "", "axis": "", "diagnoses": []}
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        body = line[1:].strip()
        m = re.match(r"<age>:\s*(\S+)", body)
        if m:
            try:
                out["age"] = float(m.group(1))
            except ValueError:
                out["age"] = None
            continue
        m = re.match(r"<sex>:\s*(\S+)", body)
        if m:
            out["sex"] = {"f": "female", "m": "male"}.get(m.group(1).lower())
            continue
        if body.lower().startswith("rhythm:"):
            out["rhythm"] = body.split(":", 1)[1].strip().rstrip(".")
        elif body.lower().startswith("electric axis"):
            out["axis"] = body.split(":", 1)[1].strip().rstrip(".")
        elif body and not body.startswith("<") and not body.lower().startswith("diagnoses"):
            out["diagnoses"].append(body.rstrip(". "))
    return out


def fetch(out: Path, limit: int | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    records = [ln.strip() for ln in fetch_bytes(BASE_URL + "RECORDS").decode().splitlines() if ln.strip()]
    if limit:
        records = records[:limit]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for n, rec in enumerate(records, 1):
        name = rec.split("/")[-1]
        files = [f"{name}.hea", f"{name}.dat"] + [f"{name}.{lead}" for lead in LEAD_FILES]
        for fname in files:
            dst = out / fname
            if dst.exists() and dst.stat().st_size > 0:
                continue
            dst.write_bytes(fetch_bytes(f"{BASE_URL}{rec.rsplit('/', 1)[0]}/{fname}"))
        if name not in manifest:
            manifest[name] = parse_header_comments((out / f"{name}.hea").read_text(errors="replace"))
        if n % 20 == 0 or n == len(records):
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
            print(f"{n}/{len(records)}", file=sys.stderr, flush=True)
    return len(records)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/ludb")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    n = fetch(Path(args.out), args.limit)
    print(f"fetched {n} records into {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
