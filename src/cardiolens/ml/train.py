"""Train / evaluate the learnable layer on local PhysioNet checkouts (TZ §17 v2).

    python -m cardiolens.ml.train \\
        --train data/ecg-arrhythmia-train --holdout data/ecg-arrhythmia-balanced \\
        --cross data/ptbxl --out configs/models/rhythm_hgb.joblib

Features are cached next to each dataset (``.ml_features.npz``) because running
Layer 1-3 on a few thousand records is the expensive part. Records present in the
hold-out set are removed from training by name. Metrics are reported for the
rules alone, the model alone and rules OR model, so the value of the model is
visible per class and never overstated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from cardiolens.ml.features import FEATURE_NAMES, features_from_result
from cardiolens.ml.model import RhythmModel, fit_model
from cardiolens.validation.metrics import score_findings

CLASSES = [
    "atrial_fibrillation", "atrial_flutter", "sinus_rhythm", "sinus_bradycardia",
    "sinus_tachycardia", "first_degree_av_block", "rbbb", "lbbb", "lvh", "low_qrs_voltage",
    "long_qt", "st_depression", "st_elevation", "left_axis_deviation", "right_axis_deviation",
    "complete_av_block", "wpw_preexcitation",
]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cases(root: Path):
    from cardiolens.validation import build_ecg_arrhythmia_dataset, build_ptbxl_dataset

    if (root / "manifest.json").exists() and (root / "ptbxl_database.csv").exists():
        return build_ptbxl_dataset(root, limit=100000)
    return build_ecg_arrhythmia_dataset(root, limit=100000)


def extract(root: Path) -> dict:
    """Features, gold labels, rule predictions and names for one dataset (cached)."""
    cache = root / ".ml_features.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in ("x", "gold", "rules", "names")}
    from cardiolens.layer6_interpret import StubProvider
    from cardiolens.pipeline import run_pipeline

    xs, golds, rules, names = [], [], [], []
    cases = _cases(root)
    for i, c in enumerate(cases, 1):
        r = run_pipeline(c.signal, mode="A", provider=StubProvider())
        xs.append(features_from_result(r))
        golds.append(json.dumps(sorted(c.gold_findings)))
        rules.append(json.dumps(sorted(r.findings.codes())))
        names.append(c.name)
        if i % 200 == 0:
            _log(f"[ml] {root.name}: {i}/{len(cases)}")
    out = {"x": np.vstack(xs), "gold": np.array(golds), "rules": np.array(rules), "names": np.array(names)}
    np.savez(cache, **out)
    return out


def _targets(gold: np.ndarray) -> dict[str, np.ndarray]:
    sets = [set(json.loads(g)) for g in gold]
    return {c: np.array([c in s for s in sets]) for c in CLASSES}


def evaluate(model: RhythmModel, data: dict, threshold: float) -> dict:
    sets_gold = [set(json.loads(g)) for g in data["gold"]]
    sets_rules = [set(json.loads(g)) for g in data["rules"]]
    probs = [model.predict_proba(row) for row in data["x"]]
    sets_ml = [{c for c, p in pr.items() if p >= threshold} for pr in probs]
    sets_or = [a | b for a, b in zip(sets_rules, sets_ml)]
    out = {}
    for label, preds in (("rules", sets_rules), ("model", sets_ml), ("rules_or_model", sets_or)):
        m = score_findings(preds, sets_gold, model.classes)
        out[label] = {c: {"sens": v.sensitivity, "spec": v.specificity, "n": v.support} for c, v in m.items()}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the CardioLens learnable layer")
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--cross", nargs="*", default=[], help="cross-dataset test sets")
    ap.add_argument("--out", default="configs/models/rhythm_hgb.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-positives", type=int, default=20)
    ap.add_argument("--report", default=None, help="write metrics JSON here")
    args = ap.parse_args(argv)

    holdouts = [extract(Path(p)) for p in args.holdout]
    exclude = {n for h in holdouts for n in h["names"].tolist()}
    train_parts = [extract(Path(p)) for p in args.train]
    x = np.vstack([t["x"] for t in train_parts])
    gold = np.concatenate([t["gold"] for t in train_parts])
    names = np.concatenate([t["names"] for t in train_parts])
    keep = np.array([n not in exclude for n in names])
    x, gold = x[keep], gold[keep]
    _log(f"[ml] training on {len(x)} records ({int((~keep).sum())} removed as hold-out duplicates)")

    model = fit_model(x, _targets(gold), FEATURE_NAMES, min_positives=args.min_positives)
    model.meta = {"train_records": int(len(x)), "train_sets": args.train, "classes": model.classes,
                  "threshold": args.threshold, "reports": {}}
    for label, roots in (("holdout", args.holdout), ("cross", args.cross)):
        for root in roots:
            data = extract(Path(root)) if root not in args.holdout else holdouts[args.holdout.index(root)]
            rep = evaluate(model, data, args.threshold)
            model.meta["reports"][f"{label}:{Path(root).name}"] = rep
            _log(f"[ml] {label} {Path(root).name}: {len(data['x'])} records")
            for c in model.classes:
                r, m_, o = rep["rules"][c], rep["model"][c], rep["rules_or_model"][c]
                _log(f"     {c:22s} n={r['n']:4d} rules {r['sens']}/{r['spec']}  model {m_['sens']}/{m_['spec']}  or {o['sens']}/{o['spec']}")
    model.save(args.out)
    if args.report:
        Path(args.report).write_text(json.dumps(model.meta, indent=2), encoding="utf-8")
    _log(f"[ml] saved {args.out} ({Path(args.out).stat().st_size // 1024} KB), classes={model.classes}")


if __name__ == "__main__":
    main()
