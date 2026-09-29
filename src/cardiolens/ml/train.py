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

from cardiolens.ml.features import FEATURE_NAMES, FEATURE_SET, features_from_result
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
        if "feature_set" in z.files and str(z["feature_set"]) == FEATURE_SET and z["x"].shape[1] == len(FEATURE_NAMES):
            return {k: z[k] for k in ("x", "gold", "rules", "names")}
        _log(f"[ml] {root.name}: cache is from another feature set, re-extracting")
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
    np.savez(cache, feature_set=np.array(FEATURE_SET), **out)
    return out


def split_ptbxl_folds(data: dict, root: Path, train_folds: set[int]) -> tuple[dict, dict]:
    """Split an extracted PTB-XL set by ``strat_fold`` (ptbxl_database.csv) into
    (train part, test part) — the folds are PTB-XL's own patient-disjoint split."""
    import csv

    fold = {}
    with (root / "ptbxl_database.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fold[f"ptbxl_{row['ecg_id']}"] = int(float(row["strat_fold"]))
    is_train = np.array([fold.get(n, 0) in train_folds for n in data["names"]])
    sub = lambda mask: {k: v[mask] for k, v in data.items()}
    return sub(is_train), sub(~is_train)


def _subset_names(data: dict, exclude: set[str]) -> dict:
    keep = np.array([n not in exclude for n in data["names"]])
    return {k: v[keep] for k, v in data.items()}


def transfer_report(sources: dict[str, dict], threshold: float, min_positives: int) -> dict:
    """Train on each source, test on every other: the cross-database gap per class."""
    out = {}
    for name, data in sources.items():
        model = fit_model(data["x"], _targets(data["gold"]), FEATURE_NAMES, min_positives=min_positives)
        for other, test in sources.items():
            if other == name:
                continue
            rep = evaluate(model, _subset_names(test, set(data["names"].tolist())), threshold)
            out[f"{name}->{other}"] = rep
            _log(f"[ml] transfer {name} -> {other}:")
            for c in model.classes:
                r, m_ = rep["rules"][c], rep["model"][c]
                _log(f"     {c:22s} n={r['n']:4d} rules {r['sens']}/{r['spec']}  model {m_['sens']}/{m_['spec']}")
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


def suggest_classes(model: RhythmModel, min_spec: float = 0.95) -> list[str]:
    """Classes the model may *suggest* on its own: precise on every evaluation set
    and at least as sensitive as the rules on the cross-dataset sets."""
    reports = model.meta.get("reports", {})
    if not reports:
        return []
    out = []
    for c in model.classes:
        ok = True
        for name, rep in reports.items():
            mm, rr = rep["model"][c], rep["rules"][c]
            if mm["spec"] is None or mm["spec"] < min_spec:
                ok = False
            if name.startswith("cross") and mm["sens"] is not None and rr["sens"] is not None and mm["sens"] < rr["sens"]:
                ok = False
        if ok:
            out.append(c)
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
    ap.add_argument("--ptbxl-train-folds", default="1-8",
                    help="a PTB-XL root given in --train is split by strat_fold: these folds train, the rest are a hold-out")
    ap.add_argument("--transfer", nargs="*", default=[],
                    help="named sources 'label=root' for a train-on-one/test-on-other transfer report")
    args = ap.parse_args(argv)

    lo, hi = (int(v) for v in args.ptbxl_train_folds.split("-"))
    train_folds = set(range(lo, hi + 1))
    holdouts = [extract(Path(p)) for p in args.holdout]
    holdout_names = list(args.holdout)
    train_parts = []
    for p in args.train:
        data = extract(Path(p))
        if (Path(p) / "ptbxl_database.csv").exists():
            data, rest = split_ptbxl_folds(data, Path(p), train_folds)
            holdouts.append(rest)
            holdout_names.append(f"{Path(p).name}-folds-{max(train_folds) + 1}-10")
            _log(f"[ml] {Path(p).name}: folds {args.ptbxl_train_folds} train ({len(data['x'])}), rest hold-out ({len(rest['x'])})")
        train_parts.append(data)
    exclude = {n for h in holdouts for n in h["names"].tolist()}
    x = np.vstack([t["x"] for t in train_parts])
    gold = np.concatenate([t["gold"] for t in train_parts])
    names = np.concatenate([t["names"] for t in train_parts])
    keep = np.array([n not in exclude for n in names])
    x, gold = x[keep], gold[keep]
    _log(f"[ml] training on {len(x)} records ({int((~keep).sum())} removed as hold-out duplicates)")

    model = fit_model(x, _targets(gold), FEATURE_NAMES, min_positives=args.min_positives)
    model.meta.update({"train_records": int(len(x)), "train_sets": args.train, "classes": model.classes,
                       "threshold": args.threshold, "reports": {}})
    evals = [("holdout", Path(n).name, d) for n, d in zip(holdout_names, holdouts)]
    evals += [("cross", Path(root).name, extract(Path(root))) for root in args.cross]
    for label, name, data in evals:
        rep = evaluate(model, data, args.threshold)
        model.meta["reports"][f"{label}:{name}"] = rep
        _log(f"[ml] {label} {name}: {len(data['x'])} records")
        for c in model.classes:
            r, m_, o = rep["rules"][c], rep["model"][c], rep["rules_or_model"][c]
            _log(f"     {c:22s} n={r['n']:4d} rules {r['sens']}/{r['spec']}  model {m_['sens']}/{m_['spec']}  or {o['sens']}/{o['spec']}")
    if args.transfer:
        sources = {}
        for item in args.transfer:
            label, root = item.split("=", 1)
            sources[label] = extract(Path(root))
        model.meta["transfer"] = transfer_report(sources, args.threshold, args.min_positives)
    model.meta["suggest_classes"] = suggest_classes(model)
    _log(f"[ml] suggestion-eligible classes (spec >= 0.95 on hold-out and cross, sens >= rules on cross): "
         f"{model.meta['suggest_classes']}")
    model.save(args.out)
    if args.report:
        Path(args.report).write_text(json.dumps(model.meta, indent=2), encoding="utf-8")
    _log(f"[ml] saved {args.out} ({Path(args.out).stat().st_size // 1024} KB), classes={model.classes}")


if __name__ == "__main__":
    main()
