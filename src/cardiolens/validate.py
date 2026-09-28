"""Validation runner (TZ §16).

    python -m cardiolens.validate                        # synthetic labelled dataset
    python -m cardiolens.validate --ptbxl <root>         # real PTB-XL (local checkout)
    python -m cardiolens.validate --ecg-arrhythmia <root>  # Chapman/Ningbo (local checkout)
"""

from __future__ import annotations

import argparse
import json

from cardiolens.validation import (
    build_ecg_arrhythmia_dataset,
    build_ptbdb_dataset,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    filter_by_age,
    ptbdb_report_lines,
    run_validation,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="CardioLens validation")
    ap.add_argument("--ptbxl", help="path to a local PTB-XL checkout")
    ap.add_argument("--ptbdb", help="path to a local PTB Diagnostic DB (fetch_ptbdb)")
    ap.add_argument("--max-days", type=int, default=3,
                    help="PTBDB: MI ECGs recorded at most this many days after the infarction are STEMI gold")
    ap.add_argument("--ecg-arrhythmia",
                    help="path to a local PhysioNet ecg-arrhythmia (Chapman/Ningbo) checkout")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--age-min", type=float, help="keep patients at least this old (years)")
    ap.add_argument("--age-max", type=float, help="keep patients at most this old (years)")
    args = ap.parse_args()

    if args.ptbxl:
        cases = build_ptbxl_dataset(args.ptbxl, limit=args.limit)
        source = f"PTB-XL ({args.ptbxl})"
    elif args.ptbdb:
        cases = build_ptbdb_dataset(args.ptbdb, limit=args.limit, max_days=args.max_days)
        source = f"PTBDB ({args.ptbdb}, STEMI gold = MI <= {args.max_days} d)"
    elif args.ecg_arrhythmia:
        cases = build_ecg_arrhythmia_dataset(args.ecg_arrhythmia, limit=args.limit)
        source = f"ecg-arrhythmia ({args.ecg_arrhythmia})"
    else:
        cases = build_synthetic_dataset()
        source = "synthetic (offline)"

    if args.age_min is not None or args.age_max is not None:
        cases = filter_by_age(cases, args.age_min, args.age_max)
        source += f" [age {args.age_min or 0}-{args.age_max or '∞'}]"
    report = run_validation(cases)
    print(f"=== CardioLens validation — {source} ===")
    print(f"cases: {report.n_cases}")
    print(f"macro sensitivity: {report.macro_sensitivity}")
    print(f"PR MAE: {report.pr_mae_ms} ms | QT MAE: {report.qt_mae_ms} ms")
    print("\nper-class (sensitivity / specificity / support):")
    for lbl, m in report.per_class.items():
        print(f"  {lbl:24s} sens={m.sensitivity} spec={m.specificity} n={m.support}")
    print("\nfamilies (any-of prediction vs any-of gold):")
    for lbl, m in report.families.items():
        print(f"  {lbl:24s} sens={m.sensitivity} spec={m.specificity} n={m.support}")
    if args.ptbdb:
        print("\nPTBDB STEMI report (alert = st_elevation; any = descriptor family):")
        for line in ptbdb_report_lines(cases, report):
            print("  " + line)
    print("\nJSON:")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
