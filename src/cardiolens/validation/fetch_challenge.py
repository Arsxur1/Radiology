"""Fetch a subset of a PhysioNet/CinC Challenge 2020 training source (CPSC-2018 etc.).

The 2020 challenge training data (``challenge-2020/1.0.2/training/<source>/gN/``)
uses the same WFDB-header-plus-``#Dx:`` layout as the ecg-arrhythmia database, so
``build_ecg_arrhythmia_dataset`` reads the result unchanged. Records are selected
by SNOMED code with a per-code cap; headers are cached so re-runs are cheap.

    python -m cardiolens.validation.fetch_challenge --source cpsc_2018 \
        --codes 164930006:all 429622005:300 426783006:300 --out data/cpsc2018

Datasets are licensed (CC BY 4.0 for the challenge data) and never committed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from cardiolens.validation.fetch_physionet import fetch_bytes, parse_age_from_header

BASE = "https://physionet.org/files/challenge-2020/1.0.2/training"


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def list_dir(url: str) -> list[str]:
    html = fetch_bytes(url).decode("utf-8", "replace")
    return [m for m in re.findall(r'href="([^"?]+)"', html) if m not in ("../", "..")]


def parse_dx(text: str) -> list[str]:
    m = re.search(r"^#\s*Dx:\s*(.+)$", text, flags=re.MULTILINE)
    return [c.strip() for c in m.group(1).split(",") if c.strip()] if m else []


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fetch a Challenge-2020 training subset")
    ap.add_argument("--source", default="cpsc_2018")
    ap.add_argument("--codes", nargs="+", required=True, help="SNOMED:cap (cap 'all' or an int)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-records", type=int, default=100000)
    args = ap.parse_args(argv)

    out = Path(args.out)
    hdr_dir = out / ".headers"
    hdr_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    for item in args.codes:
        code, cap = item.split(":")
        caps[code] = None if cap == "all" else int(cap)

    # 1. all headers (cached)
    groups = [g.rstrip("/") for g in list_dir(f"{BASE}/{args.source}/") if g.endswith("/")]
    _log(f"[fetch] {args.source}: {len(groups)} groups")
    headers: dict[str, str] = {}
    for g in groups:
        names = [n for n in list_dir(f"{BASE}/{args.source}/{g}/") if n.endswith(".hea")]
        for n in names:
            p = hdr_dir / g / n
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(fetch_bytes(f"{BASE}/{args.source}/{g}/{n}"))
            headers[f"{g}/{n[:-4]}"] = p.read_text(encoding="utf-8", errors="replace")
        _log(f"[fetch] headers {g}: {len(names)} (total {len(headers)})")

    # 2. selection
    counts = dict.fromkeys(caps, 0)
    chosen: list[str] = []
    for rec in sorted(headers):
        dx = set(parse_dx(headers[rec]))
        for code, cap in caps.items():
            if code in dx and (cap is None or counts[code] < cap):
                counts[code] += 1
                chosen.append(rec)
                break
        if len(chosen) >= args.max_records:
            break
    _log(f"[fetch] selected {len(chosen)} records: {counts}")

    # 3. signals
    manifest = []
    for i, rec in enumerate(chosen):
        g, name = rec.split("/")
        dst = out / "WFDBRecords" / g
        dst.mkdir(parents=True, exist_ok=True)
        if not (dst / f"{name}.mat").exists():
            (dst / f"{name}.hea").write_text(headers[rec], encoding="utf-8")
            (dst / f"{name}.mat").write_bytes(fetch_bytes(f"{BASE}/{args.source}/{g}/{name}.mat"))
        manifest.append({"record": rec, "dx": parse_dx(headers[rec]), "age": parse_age_from_header(headers[rec])})
        if i % 50 == 0:
            _log(f"[fetch] signals {i}/{len(chosen)}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    _log(f"[fetch] done: {len(manifest)} records in {out}")


if __name__ == "__main__":
    main()
