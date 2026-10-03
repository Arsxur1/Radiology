"""De-warp from the grid texture (D64): local grid frequencies -> one homography."""

import cv2
import numpy as np

from cardiolens.layer1_ingestion import digitize as D
from cardiolens.layer1_ingestion.synthetic import generate_synthetic_ecg


def _page(tmp_path, **degrade):
    sig = generate_synthetic_ecg(heart_rate_bpm=72)
    page = tmp_path / "page.png"
    D.render_ecg_page(sig, page, px_per_mm=6, calibration_pulse=True)
    if degrade:
        page = D.degrade_image(page, tmp_path / "deg.png", seed=1, **degrade)
    return cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)


def test_local_frequencies_find_the_grid_pitch_and_axes(tmp_path):
    gray = _page(tmp_path)
    samples = D._local_grid_frequencies(gray)
    assert len(samples) >= D.GRID_TEXTURE_MIN_TILES
    pitch = np.array([1.0 / np.linalg.norm(s[2]) for s in samples] + [1.0 / np.linalg.norm(s[3]) for s in samples])
    assert abs(np.median(pitch) - 6.0) < 0.3                 # 6 px/mm, 1 mm grid
    ang1 = np.degrees(np.arctan2([s[2][1] for s in samples], [s[2][0] for s in samples]))
    ang2 = np.degrees(np.arctan2([s[3][1] for s in samples], [s[3][0] for s in samples]))
    assert abs(np.median(ang1)) < 1.5 and abs(np.median(ang2) - 90) < 1.5


def test_a_perspective_photo_is_rectified_to_a_square_grid(tmp_path):
    gray = _page(tmp_path, perspective=0.035, vignette=0.6, rotation_deg=-2.5)
    out = D._dewarp_by_grid_texture(gray)
    assert out is not None
    px, py = D.estimate_grid_pitch_xy(out)
    assert px and py and 0.97 < px / py < 1.03
    hm, _p, rms = D._fit_grid_homography(D._local_grid_frequencies(gray))
    assert rms < D.GRID_TEXTURE_MAX_RMS
    assert np.linalg.det(hm[:2, :2]) > 0                       # no reflection


def test_a_rotation_alone_is_left_to_the_deskew(tmp_path):
    gray = _page(tmp_path, rotation_deg=1.5, blur_px=1, jpeg_quality=60)
    assert D._dewarp_by_grid_texture(gray) is None             # affine: no projective part


def test_a_page_without_grid_texture_gives_no_fit():
    blank = np.full((600, 900), 255, np.uint8)
    assert D._dewarp_by_grid_texture(blank) is None
