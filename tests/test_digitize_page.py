"""12-lead page digitisation (TZ §4, §17 "грязные фото"): render -> degrade -> recover."""

import numpy as np
import pytest

from cardiolens.layer1_ingestion import generate_synthetic_ecg
from cardiolens.layer1_ingestion.digitize import (
    LAYOUT_3X4,
    PANEL_SECONDS,
    degrade_image,
    digitize_page,
    digitize_page_to_signal,
    estimate_skew,
    render_ecg_page,
)
from cardiolens.pipeline import run_pipeline

DEGRADATIONS = {
    "rotation": {"rotation_deg": 3.0},
    "blur": {"blur_px": 2},
    "jpeg": {"jpeg_quality": 40},
    "shadow": {"shadow": 0.5},
    "salt": {"salt_noise": 0.005},
    "combo": {"rotation_deg": -2.0, "blur_px": 1, "jpeg_quality": 40, "shadow": 0.4, "salt_noise": 0.003},
}


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    b = np.interp(np.linspace(0, 1, len(a)), np.linspace(0, 1, len(b)), b)
    return float(np.corrcoef(a, b)[0, 1])


def _per_lead_r(sig, traces) -> list[float]:
    seg = int(PANEL_SECONDS * sig.sampling_rate_hz)
    return [
        _corr(sig.lead(lead)[col * seg:(col + 1) * seg], traces[lead])
        for row in LAYOUT_3X4 for col, lead in enumerate(row)
    ]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    sig = generate_synthetic_ecg(heart_rate_bpm=72, duration_s=10.0, seed=3)
    path = str(tmp_path_factory.mktemp("page") / "page.png")
    render_ecg_page(sig, path)
    return sig, path


def test_clean_page_roundtrip_all_leads(page):
    sig, path = page
    traces = digitize_page(path)
    assert set(traces) == {lead for row in LAYOUT_3X4 for lead in row} | {"II_rhythm"}
    rs = _per_lead_r(sig, traces)
    assert min(rs) > 0.85 and np.mean(rs) > 0.9
    assert digitize_page.last_coverage > 0.98
    # Rhythm strip covers the full 10 s of lead II.
    assert _corr(sig.lead("II"), traces["II_rhythm"]) > 0.9


@pytest.mark.parametrize("name", sorted(DEGRADATIONS))
def test_degraded_page_roundtrip(page, tmp_path, name):
    sig, path = page
    dst = degrade_image(path, tmp_path / f"{name}.png", **DEGRADATIONS[name])
    traces = digitize_page(dst)
    rs = _per_lead_r(sig, traces)
    assert min(rs) > 0.8, (name, rs)
    assert np.mean(rs) > 0.88, (name, np.mean(rs))


def test_skew_is_estimated_and_undone(page, tmp_path):
    import cv2

    _sig, path = page
    dst = degrade_image(path, tmp_path / "rot.png", rotation_deg=3.0)
    # Sign follows the image y-axis (down); digitize_page feeds it straight back to warpAffine.
    assert abs(estimate_skew(cv2.imread(dst, cv2.IMREAD_GRAYSCALE))) == pytest.approx(3.0, abs=0.5)


def test_page_to_signal_windows_and_pipeline(page):
    sig, path = page
    dig = digitize_page_to_signal(path)
    assert dig.source == "image" and dig.provenance.digitized
    assert dig.provenance.confidence >= 0.95
    windows = dig.provenance.lead_windows
    assert windows["II"] == (0.0, 10.0)                # rhythm strip
    assert windows["V6"] == (7.5, 10.0)                # last column
    # Outside its window a panel lead is silent, inside it carries the trace.
    v6 = dig.lead("V6")
    fs = dig.sampling_rate_hz
    assert np.all(v6[: int(7.4 * fs)] == 0)
    assert np.abs(v6[int(7.5 * fs):]).max() > 200

    result = run_pipeline(dig, mode="A")
    assert result.measurements.heart_rate_bpm == pytest.approx(72, abs=3)
    assert result.measurements.axes.qrs_deg == pytest.approx(
        run_pipeline(sig, mode="A").measurements.axes.qrs_deg, abs=15)


def test_stemi_page_keeps_critical_finding(tmp_path):
    sig = generate_synthetic_ecg(heart_rate_bpm=80, duration_s=10.0, st_elevation_uv=300,
                                 st_leads=["V2", "V3", "V4"], seed=5)
    path = str(tmp_path / "stemi.png")
    render_ecg_page(sig, path)
    dst = degrade_image(path, tmp_path / "stemi_photo.jpg", rotation_deg=2.0, blur_px=1,
                        jpeg_quality=40, shadow=0.3, salt_noise=0.002)
    result = run_pipeline(digitize_page_to_signal(dst), mode="A")
    assert "st_elevation" in result.findings.codes()
