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


def _recovery(img_path, lead):
    """(correlation, amplitude gain) of the digitised trace vs the original."""
    rec = resample(digitize_image(img_path).lead("II"), len(lead))
    return float(np.corrcoef(lead, rec)[0, 1]), float(np.std(rec) / np.std(lead))


def test_red_grid_is_not_mistaken_for_trace(tmp_path):
    """ecg-image-kit style red grid: dark in grayscale, bright in the R channel."""
    import cv2

    orig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=6)
    lead = orig.lead("II")
    img = str(tmp_path / "ecg.png")
    render_ecg_image(lead, orig.sampling_rate_hz, img)
    im = cv2.imread(img)
    grid = np.zeros(im.shape[:2], dtype=bool)
    grid[:, ::8] = grid[::8, :] = True
    trace = im.max(axis=2) < 100
    # BGR red, grayscale ~ 95 (below the dark threshold); trace drawn on top.
    im[grid & ~trace] = (60, 60, 220)
    cv2.imwrite(img, im)
    corr, gain = _recovery(img, lead)
    assert corr > 0.9
    assert 0.8 < gain < 1.2  # amplitude not flattened toward the grid


def test_neighbouring_trace_in_crop_is_not_averaged_in(tmp_path):
    """A trace of the adjacent lead entering the crop must not pull the output."""
    import cv2

    orig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=6)
    lead = orig.lead("II")
    img = str(tmp_path / "ecg.png")
    render_ecg_image(lead, orig.sampling_rate_hz, img)
    im = cv2.imread(img)
    h, w = im.shape[:2]
    xs = np.arange(w)
    ys = (h // 8 + 6 * np.sin(xs / 15.0)).astype(int)  # foreign trace near the top
    cv2.polylines(im, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (20, 20, 20), 2)
    cv2.imwrite(img, im)
    corr, gain = _recovery(img, lead)
    assert corr > 0.9
    assert 0.8 < gain < 1.2  # amplitude not flattened toward the grid


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


def test_shaded_image_uses_adaptive_mask_and_escalates(tmp_path):
    """Shading/creases (large dark areas): trace still recovered, but the image
    is flagged degraded (capped provenance confidence) and the report escalates."""
    import cv2

    orig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=6)
    lead = orig.lead("II")
    img = str(tmp_path / "ecg.png")
    render_ecg_image(lead, orig.sampling_rate_hz, img)
    im = cv2.imread(img).astype(np.float32)
    h, w = im.shape[:2]
    shade = np.ones((h, w), np.float32)
    shade[:, w // 3: 2 * w // 3] = 0.35  # a dark crease band across the strip
    im = (im * shade[:, :, None]).astype(np.uint8)
    cv2.imwrite(img, im)

    dig = digitize_image(img)
    assert dig.provenance.confidence <= 0.5
    corr, _ = _recovery(img, lead)
    assert corr > 0.8

    result = run_pipeline(dig, mode="B")
    assert any("provenance confidence" in line for line in result.audit_log)
    assert result.report.escalation_reason


def test_clean_digitized_image_is_not_escalated_for_provenance(tmp_path):
    orig = generate_synthetic_ecg(heart_rate_bpm=75, duration_s=6)
    img = str(tmp_path / "ecg.png")
    render_ecg_image(orig.lead("II"), orig.sampling_rate_hz, img)
    result = run_pipeline(digitize_image(img), mode="B")
    assert not any("provenance confidence" in line for line in result.audit_log)
