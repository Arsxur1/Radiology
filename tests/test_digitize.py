"""Paper-ECG digitisation tests (TZ §4) — render/digitise round-trip."""

import numpy as np
import pytest
from scipy.signal import resample

from cardiolens.layer1_ingestion import (
    digitize_image,
    generate_synthetic_ecg,
    render_ecg_image,
)
from cardiolens.pipeline import run_pipeline


def test_render_digitize_roundtrip_preserves_morphology(tmp_path):
    orig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=6)
    lead = orig.lead("II")
    img = str(tmp_path / "ecg.png")
    render_ecg_image(lead, orig.sampling_rate_hz, img,
                     px_per_mm=8, mm_per_s=25, mm_per_mv=10)

    dig = digitize_image(img, px_per_mm=8, mm_per_s=25, mm_per_mv=10)
    assert dig.source == "image"
    assert dig.provenance.digitized is True
    assert dig.provenance.confidence >= 0.95  # clean render -> high coverage

    recovered = resample(dig.lead("II"), len(lead))
    r = float(np.corrcoef(lead, recovered)[0, 1])
    assert r > 0.9  # morphology well preserved


def test_digitized_signal_recovers_heart_rate(tmp_path):
    orig = generate_synthetic_ecg(heart_rate_bpm=88, duration_s=6)
    img = str(tmp_path / "ecg.png")
    render_ecg_image(orig.lead("II"), orig.sampling_rate_hz, img)
    result = run_pipeline(digitize_image(img), mode="A")
    assert result.measurements.heart_rate_bpm == pytest.approx(88, abs=4)


def test_digitize_missing_file():
    with pytest.raises(FileNotFoundError):
        digitize_image("/nonexistent/does_not_exist.png")


def test_digitize_blank_image_raises(tmp_path):
    import cv2

    blank = np.full((100, 200), 255, dtype=np.uint8)
    p = str(tmp_path / "blank.png")
    cv2.imwrite(p, blank)
    with pytest.raises(ValueError):
        digitize_image(p)
