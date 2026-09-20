"""Train the transparent LDA beat classifier on a local MIT-BIH checkout.

    python -m cardiolens.train_beats --mitdb <root> [--out configs/beat_lda.json]

Reads WFDB records + ``.atr`` beat annotations, extracts the same scale-invariant
features the runtime uses (``arrhythmia.extract_beat_features``), fits a 3-class
LDA (N/S/V) and writes its coefficients as JSON. Network to PhysioNet is not
needed at runtime; the committed JSON is what the pipeline loads. See DECISIONS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cardiolens.arrhythmia.beats import FEATURES, extract_beat_features

# AAMI beat-symbol -> class; non-beat annotation symbols (+ ~ | " etc.) excluded.
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "AaJS"},
        **{s: "V" for s in "VE"}}
CLASSES = ["N", "S", "V"]


def _record_xy(rec: Path, fs_bandpass=(0.5, 40.0)):
    import wfdb
    from scipy import signal as sp

    r = wfdb.rdrecord(str(rec))
    ann = wfdb.rdann(str(rec), "atr")
    fs = r.fs
    lead = r.p_signal[:, 0].astype(float)
    b, a = sp.butter(3, [fs_bandpass[0] / (fs / 2), fs_bandpass[1] / (fs / 2)], btype="band")
    lead = sp.filtfilt(b, a, lead)
    beats = [(s, AAMI[sym]) for s, sym in zip(ann.sample, ann.symbol) if sym in AAMI]
    if len(beats) < 4:
        return None, None
    rpeaks = [s for s, _ in beats]
    labels = [lab for _, lab in beats]
    X, idx = extract_beat_features(lead, rpeaks, fs)
    if len(X) == 0:
        return None, None
    return X, np.array([labels[i] for i in idx])


def fit_lda(X: np.ndarray, Y: np.ndarray) -> dict:
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xz = (X - mu) / sd
    means, priors = {}, {}
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c in CLASSES:
        Xc = Xz[Y == c]
        means[c] = Xc.mean(0)
        priors[c] = len(Xc) / len(Xz)
        Sw += (Xc - means[c]).T @ (Xc - means[c])
    Sw /= len(Xz)
    inv_cov = np.linalg.pinv(Sw)
    return {
        "classes": CLASSES,
        "features": FEATURES,
        "mu": mu.tolist(), "sd": sd.tolist(),
        "means": {c: means[c].tolist() for c in CLASSES},
        "inv_cov": inv_cov.tolist(),
        "priors": priors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train MIT-BIH beat classifier")
    ap.add_argument("--mitdb", required=True, help="local MIT-BIH checkout dir")
    ap.add_argument("--out", default="configs/beat_lda.json")
    ap.add_argument("--records", help="space-separated record ids (default: all in dir)")
    args = ap.parse_args()

    root = Path(args.mitdb)
    if args.records:
        recs = [root / r for r in args.records.split()]
    else:
        recs = sorted({p.with_suffix("") for p in root.glob("*.atr")})

    Xs, Ys = [], []
    for rec in recs:
        X, Y = _record_xy(rec)
        if X is not None:
            Xs.append(X)
            Ys.append(Y)
    X = np.vstack(Xs)
    Y = np.concatenate(Ys)
    keep = np.isin(Y, CLASSES)
    model = fit_lda(X[keep], Y[keep])
    Path(args.out).write_text(json.dumps(model, indent=2))
    from collections import Counter
    print(f"trained on {int(keep.sum())} beats from {len(Xs)} records: {Counter(Y[keep])}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
