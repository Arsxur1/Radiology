"""Validation runner (TZ §16).

    python -m cardiolens.validate                        # synthetic labelled dataset
    python -m cardiolens.validate --ptbxl <root>         # real PTB-XL (local checkout)
    python -m cardiolens.validate --ecg-arrhythmia <root>  # Chapman/Ningbo (local checkout)

Each source scores the finding codes its label map can actually supply
(``PTBXL_SCORED_LABELS`` / ``ECG_ARRHYTHMIA_SCORED_LABELS``); pass ``--labels``
to override with an explicit comma-separated set.
"""

from __future__ import annotations

import argparse
import json

from cardiolens.validation import (
    ECG_ARRHYTHMIA_SCORED_LABELS,
    PTBXL_SCORED_LABELS,
    SCORED_LABELS,
    build_ecg_arrhythmia_dataset,
    build_ptbxl_dataset,
    build_synthetic_dataset,
    run_validation,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="CardioLens validation")
    ap.add_argument("--ptbxl", help="path to a local PTB-XL checkout")
    ap.add_argument("--ecg-arrhythmia",
                    help="path to a local PhysioNet ecg-arrhythmia (Chapman/Ningbo) checkout")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--labels",
                    help="comma-separated finding codes to score (overrides the "
                         "per-source default)")
    args = ap.parse_args()

    if args.ptbxl:
        cases = build_ptbxl_dataset(args.ptbxl, limit=args.limit)
        source = f"PTB-XL ({args.ptbxl})"
        labels = PTBXL_SCORED_LABELS
    elif args.ecg_arrhythmia:
        cases = build_ecg_arrhythmia_dataset(args.ecg_arrhythmia, limit=args.limit)
        source = f"ecg-arrhythmia ({args.ecg_arrhythmia})"
        labels = ECG_ARRHYTHMIA_SCORED_LABELS
    else:
        cases = build_synthetic_dataset()
        source = "synthetic (offline)"
        labels = SCORED_LABELS

    if args.labels:
        labels = [c.strip() for c in args.labels.split(",") if c.strip()]

    report = run_validation(cases, labels=labels)
    print(f"=== CardioLens validation — {source} ===")
    print(f"cases: {report.n_cases}")
    print(f"macro sensitivity: {report.macro_sensitivity}")
    print(f"PR MAE: {report.pr_mae_ms} ms | QT MAE: {report.qt_mae_ms} ms")
    print("\nper-class (sensitivity / specificity / support):")
    for lbl, m in report.per_class.items():
        print(f"  {lbl:24s} sens={m.sensitivity} spec={m.specificity} n={m.support}")
    print("\nJSON:")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
