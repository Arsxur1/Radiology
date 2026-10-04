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


def _render_page(tmp_path, *, with_pulse=True, hr=72):
    """Synthetic 12-lead page: red 1/5-mm grid, 3x4 layout + lead-II rhythm strip,
    1 mV calibration pulse per row; 8 px/mm, 25 mm/s, 10 mm/mV."""
    import cv2

    from cardiolens.layer1_ingestion.digitize import PAGE_LAYOUT_3X4_II

    ecg = generate_synthetic_ecg(heart_rate_bpm=hr, duration_s=10)
    fs, ppm = ecg.sampling_rate_hz, 8
    w, h = int(290 * ppm), int(200 * ppm)
    img = np.full((h, w, 3), 255, np.uint8)
    for x in range(0, w, ppm):
        img[:, x] = (200, 200, 255) if x % (5 * ppm) else (60, 60, 230)
    for y in range(0, h, ppm):
        img[y, :] = (200, 200, 255) if y % (5 * ppm) else (60, 60, 230)
    x_pulse, x0 = 10 * ppm, 15 * ppm
    for r, names in enumerate(PAGE_LAYOUT_3X4_II):
        base = int((40 + 45 * r) * ppm)
        if with_pulse:
            pts = [(x_pulse, base), (x_pulse, base - 10 * ppm), (x0, base - 10 * ppm), (x0, base)]
            cv2.polylines(img, [np.array(pts).reshape(-1, 1, 2)], False, (0, 0, 0), 2)
        seg_s = 10.0 / len(names)
        for k, name in enumerate(names):
            i0, i1 = int(k * seg_s * fs), int((k + 1) * seg_s * fs)
            x = ecg.lead(name)[i0:i1]
            xs = (x0 + (np.arange(len(x)) + i0) / fs * 25 * ppm).astype(int)
            ys = (base - x / 1000.0 * 10 * ppm).astype(int)
            cv2.polylines(img, [np.stack([xs, ys], 1).reshape(-1, 1, 2)], False, (0, 0, 0), 2)
    p = str(tmp_path / "page.png")
    cv2.imwrite(p, img)
    return p, ecg


def test_digitize_page_finds_layout_and_recovers_rhythm(tmp_path):
    from cardiolens.layer1_ingestion import digitize_page

    path, ecg = _render_page(tmp_path)
    page = digitize_page(path)
    assert abs(page.px_per_mm - 8.0) < 0.2
    assert set(page.leads) == {"I", "II", "III", "aVR", "aVL", "aVF",
                               "V1", "V2", "V3", "V4", "V5", "V6"}
    assert abs(page.rhythm.duration_s - 10.0) < 0.5
    hr = run_pipeline(page.rhythm, mode="A").measurements.heart_rate_bpm
    assert hr == pytest.approx(72, abs=4)
    ref = ecg.lead("V2")[1250:2500]
    got = resample(page.leads["V2"].lead("V2"), len(ref))
    assert float(np.corrcoef(ref, got)[0, 1]) > 0.9


def test_digitize_page_without_pulse_reads_rhythm_span_and_flags_it(tmp_path):
    """D78: a template with no calibration pulse is read from the 10-s rhythm-strip span
    at the standard gain, and the page is flagged so the report escalates."""
    from cardiolens.layer1_ingestion import digitize_page

    path, _ = _render_page(tmp_path, with_pulse=False)
    page = digitize_page(path)
    assert abs(page.rhythm.duration_s - 10.0) < 0.5
    assert any("no usable calibration pulse" in n for n in page.notes)
    assert any("degraded" in n for n in page.notes)
    hr = run_pipeline(page.rhythm, mode="A").measurements.heart_rate_bpm
    assert hr == pytest.approx(72, abs=4)


def test_digitize_page_refuses_page_without_pulse_or_full_rhythm_strip(tmp_path):
    """Without a pulse AND without a 10-s rhythm trace there is no time/gain reference."""
    import cv2

    from cardiolens.layer1_ingestion import digitize_page

    path, _ = _render_page(tmp_path, with_pulse=False)
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    img[int(0.75 * h):, int(0.5 * w):] = 255        # erase the second half of the rhythm strip
    cv2.imwrite(str(path), img)
    with pytest.raises(ValueError, match="layout not recognised"):
        digitize_page(path)


def test_digitize_page_deskews_rotated_photo_and_flags_it(tmp_path):
    """A page rotated by ~2° is read after deskew, and the correction caps provenance
    confidence so the report still escalates (D32)."""
    import cv2

    from cardiolens.layer1_ingestion import digitize_page

    path, _ = _render_page(tmp_path)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    cv2.imwrite(path, cv2.warpAffine(img, rot, (w, h), borderValue=(255, 255, 255)))
    page = digitize_page(path)
    assert any("deskewed" in n for n in page.notes)
    assert page.rhythm.provenance.confidence <= 0.5
    hr = run_pipeline(page.rhythm, mode="A").measurements.heart_rate_bpm
    assert hr == pytest.approx(72, abs=4)
