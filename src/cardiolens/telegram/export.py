"""Turn the reviewed records into a labelled dataset for the validation harness,
the confidence calibrator and the learnable layer.

    python -m cardiolens.telegram export            # summary + harness report on verified records
    python -m cardiolens.validate --telegram data/telegram
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from cardiolens.telegram.analyze import load_signal
from cardiolens.telegram.store import DatasetStore, Record
from cardiolens.validation.harness import LabeledCase


def build_telegram_dataset(root: str | Path, limit: int = 5000) -> list[LabeledCase]:
    """Verified records with a digitised signal -> harness cases (gold = reviewer's label)."""
    store = DatasetStore(root)
    cases: list[LabeledCase] = []
    for rec in store.all():
        if not rec.verified or not rec.signal_path or not Path(rec.signal_path).exists():
            continue
        cases.append(LabeledCase(
            signal=load_signal(rec.signal_path), gold_findings=set(rec.gold or []), name=rec.id,
            meta={"verdict": rec.verdict, "comment": rec.comment, "chat": rec.chat_title,
                  "predicted_at_review": rec.predicted, "critical": rec.critical},
        ))
    return cases[:limit]


def agreement_summary(records: list[Record]) -> dict:
    """What the reviewer said about what the pipeline said, per code and per case."""
    verified = [r for r in records if r.verified]
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    exact = 0
    for r in verified:
        pred, gold = set(r.predicted), set(r.gold or [])
        exact += pred == gold
        for c in pred & gold:
            tp[c] += 1
        for c in pred - gold:
            fp[c] += 1
        for c in gold - pred:
            fn[c] += 1
    codes = sorted(set(tp) | set(fp) | set(fn))
    per_code = {c: {"tp": tp[c], "fp": fp[c], "fn": fn[c],
                    "precision": round(tp[c] / (tp[c] + fp[c]), 3) if tp[c] + fp[c] else None,
                    "sensitivity": round(tp[c] / (tp[c] + fn[c]), 3) if tp[c] + fn[c] else None}
                for c in codes}
    return {"verified": len(verified), "exact_match": exact,
            "exact_match_rate": round(exact / len(verified), 3) if verified else None,
            "per_code": per_code}


def summary_lines(store: DatasetStore) -> list[str]:
    s = store.stats()
    a = agreement_summary(store.all())
    lines = [
        f"records: {s['total']} (analysed {s['analysed']}, digitise failed {s['digitise_failed']}, not ECG {s['not_ecg']})",
        (f"awaiting review: {s['awaiting_review']}; verified: {s['verified']} "
         f"(confirmed {s['confirmed']}, partial {s['partial']}, rejected {s['rejected']}, comment-only {s['comment']})"),
    ]
    if a["verified"]:
        lines.append(f"exact match of the finding set: {a['exact_match']}/{a['verified']} ({a['exact_match_rate']})")
        for code, m in a["per_code"].items():
            lines.append(f"  {code:24s} tp={m['tp']} fp={m['fp']} fn={m['fn']} "
                         f"precision={m['precision']} sens={m['sensitivity']}")
    return lines
