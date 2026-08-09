"""Confidence calibration tests (TZ §13, §16)."""

import random

import pytest

from cardiolens.calibration import TemperatureScaler, expected_calibration_error


def test_identity_when_unfitted():
    s = TemperatureScaler()
    assert s.apply(0.9) == pytest.approx(0.9)
    assert not s.fitted


def test_too_few_samples_stays_identity():
    s = TemperatureScaler().fit([(0.9, 1), (0.8, 0)])
    assert not s.fitted and s.temperature == 1.0


def test_fit_reduces_calibration_error_on_overconfident_data():
    # Overconfident model: reports 0.9 but only ~60% correct.
    rng = random.Random(0)
    samples = []
    for _ in range(400):
        conf = 0.9
        label = 1 if rng.random() < 0.6 else 0
        samples.append((conf, label))
    ece_before = expected_calibration_error(samples)
    s = TemperatureScaler().fit(samples)
    assert s.fitted and s.temperature > 1.0  # temperature > 1 softens overconfidence
    scaled = [(s.apply(c), y) for c, y in samples]
    ece_after = expected_calibration_error(scaled)
    assert ece_after < ece_before


def test_ece_perfect_calibration_is_zero():
    # 50% confidence, 50% correct across many samples -> low ECE.
    samples = [(0.5, i % 2) for i in range(200)]
    assert expected_calibration_error(samples) < 0.05


def test_save_load_roundtrip(tmp_path):
    p = tmp_path / "cal.json"
    TemperatureScaler(temperature=2.5, fitted=True).save(p)
    loaded = TemperatureScaler.load(p)
    assert loaded.temperature == 2.5 and loaded.fitted


def test_load_missing_is_identity(tmp_path):
    s = TemperatureScaler.load(tmp_path / "nope.json")
    assert s.temperature == 1.0 and not s.fitted
