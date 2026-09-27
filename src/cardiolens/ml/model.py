"""Per-class gradient-boosted models over the measurement features (TZ §17 v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class RhythmModel:
    """One HistGradientBoosting classifier per finding code (multi-label)."""

    classes: list[str]
    feature_names: list[str]
    estimators: dict = field(default_factory=dict)      # code -> fitted classifier
    train_support: dict = field(default_factory=dict)   # code -> positives in training
    meta: dict = field(default_factory=dict)            # datasets, sizes, held-out metrics

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        x2 = np.asarray(x, dtype=np.float64).reshape(1, -1)
        return {code: float(est.predict_proba(x2)[0, 1]) for code, est in self.estimators.items()}

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def load_model(path: str | Path) -> RhythmModel | None:
    """Load a saved model, or None when the artifact is absent (rules-only mode)."""
    import joblib

    p = Path(path)
    if not p.exists():
        return None
    obj = joblib.load(p)
    return obj if isinstance(obj, RhythmModel) else None


def fit_model(
    x: np.ndarray, y: dict[str, np.ndarray], feature_names: list[str], *,
    min_positives: int = 20, seed: int = 0,
) -> RhythmModel:
    """Fit one classifier per code with at least ``min_positives`` positives."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = RhythmModel(classes=[], feature_names=list(feature_names))
    for code, target in y.items():
        pos = int(target.sum())
        if pos < min_positives or pos > len(target) - min_positives:
            continue
        est = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, class_weight="balanced", random_state=seed,
        )
        est.fit(x, target.astype(int))
        model.estimators[code] = est
        model.train_support[code] = pos
        model.classes.append(code)
    return model
