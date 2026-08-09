"""End-to-end demo runner (TZ §18 п.4 — pipeline on one example).

Usage:
    python -m cardiolens.demo                     # synthetic normal ECG
    python -m cardiolens.demo --case hyperk       # hyperkalemia pattern
    python -m cardiolens.demo --wfdb path/to/rec  # a real WFDB/PTB-XL record
    python -m cardiolens.demo --mode B            # request autonomous mode

Writes all five output formats (TZ §11) to ``outputs/``.
"""

from __future__ import annotations

import argparse

from cardiolens.layer1_ingestion import generate_synthetic_ecg, load_wfdb
from cardiolens.outputs import format_conclusion, write_all_outputs
from cardiolens.pipeline import run_pipeline

_CASES = {
    "normal": dict(heart_rate_bpm=72),
    "brady": dict(heart_rate_bpm=44),
    "tachy": dict(heart_rate_bpm=125),
    "hyperk": dict(heart_rate_bpm=72, t_amp_uv=750, potassium=6.6),
    "stemi": dict(heart_rate_bpm=70, st_elevation_uv=250, st_leads=["II", "III", "aVF"]),
    "child": dict(heart_rate_bpm=95, age_years=6),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="CardioLens end-to-end demo")
    ap.add_argument("--case", choices=sorted(_CASES), default="normal")
    ap.add_argument("--wfdb", help="path to a WFDB record (no extension)")
    ap.add_argument("--mode", choices=["A", "B"], default=None)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    if args.wfdb:
        sig = load_wfdb(args.wfdb)
        stem = "record"
    else:
        sig = generate_synthetic_ecg(**_CASES[args.case])
        stem = args.case

    result = run_pipeline(sig, mode=args.mode)
    paths = write_all_outputs(result, args.out, stem=stem)

    print(format_conclusion(result, "ru"))
    print("\nВыходные файлы (TZ §11):")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
