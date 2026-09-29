"""Paper-ECG digitisation (TZ §4, v2 in TZ §17 "грязные фото").

Two levels:

* **Single lead** (MVP): ``render_ecg_image`` / ``digitize_image`` — a clean gridded
  strip, column-wise trace extraction.
* **Full 12-lead page** (v2): ``render_ecg_page`` draws the standard 3x4 layout
  (four 2.5-s columns: I/II/III, aVR/aVL/aVF, V1-V3, V4-V6) plus a 10-s lead-II
  rhythm strip; ``degrade_image`` produces photo-like defects for evaluation
  (rotation, blur, JPEG, shadow gradient, salt noise); ``digitize_page`` recovers
  every lead: de-skew from the grid lines (Hough), adaptive thresholding (robust
  to shading), row/column segmentation from trace projections with a
  layout-geometry fallback, per-panel column-wise extraction, then
  ``page_to_signal`` builds a 10-s ``SignalRepresentation`` whose
  ``provenance.lead_windows`` says where each lead really has data.

Calibration (px/mm, mm/s, mm/mV) is still passed in; calibration-pulse detection
remains a TODO. ``provenance.confidence`` is the mean column coverage over the
panels, so a bad photo yields a low-confidence result rather than an invented one
(TZ §0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import (
    STANDARD_12_LEADS,
    Provenance,
    SignalRepresentation,
)

# Standard 3x4 layout: row -> leads, column c covers [2.5c, 2.5(c+1)) s.
LAYOUT_3X4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
RHYTHM_LEAD = "II"
PANEL_SECONDS = 2.5
PAGE_MARGIN_MM = 8.0
ROW_HEIGHT_MM = 32.0
LABEL_OFFSET_MM = 2.0
_GRID_LIGHT = (230, 220, 220)
_GRID_DARK = (200, 180, 180)
_TRACE = (20, 20, 20)


# ----------------------------------------------------------------------------
# Single lead (MVP)
# ----------------------------------------------------------------------------
def _draw_grid(img: np.ndarray, px_per_mm: int) -> None:
    import cv2

    h, w = img.shape[:2]
    for x in range(0, w, px_per_mm):
        cv2.line(img, (x, 0), (x, h), _GRID_LIGHT, 1)
    for y in range(0, h, px_per_mm):
        cv2.line(img, (0, y), (w, y), _GRID_LIGHT, 1)
    for x in range(0, w, px_per_mm * 5):
        cv2.line(img, (x, 0), (x, h), _GRID_DARK, 1)
    for y in range(0, h, px_per_mm * 5):
        cv2.line(img, (0, y), (w, y), _GRID_DARK, 1)


def _draw_trace(img: np.ndarray, sig_uv: np.ndarray, fs: float, x0: int, y_center: int,
                px_per_s: float, px_per_uv: float) -> None:
    import cv2

    n = len(sig_uv)
    xs = (x0 + np.arange(n) / fs * px_per_s).astype(int)
    ys = (y_center - np.asarray(sig_uv, dtype=np.float64) * px_per_uv).astype(int)
    ys = np.clip(ys, 0, img.shape[0] - 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=_TRACE, thickness=2)


def render_ecg_image(
    signal_uv: np.ndarray,
    sampling_rate_hz: float,
    path: str | Path,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    px_per_mm: int = 8,
    lead_name: str = "II",
) -> str:
    """Render one lead to a clean gridded PNG (a synthetic 'paper' strip)."""
    import cv2

    sig = np.asarray(signal_uv, dtype=np.float64)
    px_per_s = px_per_mm * mm_per_s
    px_per_uv = px_per_mm * mm_per_mv / 1000.0
    width = max(1, int(round(len(sig) / sampling_rate_hz * px_per_s)))
    height = int(round(30 * mm_per_mv / 10.0 * px_per_mm))
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
    _draw_trace(img, sig, sampling_rate_hz, 0, height // 2, px_per_s, px_per_uv)
    cv2.imwrite(str(path), img)
    return str(path)


def _trace_rows(mask: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    """Per-column median row of trace pixels inside a box (NaN where none)."""
    sub = mask[y0:y1, x0:x1]
    out = np.full(x1 - x0, np.nan)
    for i in range(x1 - x0):
        rows = np.flatnonzero(sub[:, i])
        if rows.size:
            out[i] = y0 + float(np.median(rows))
    return out


def _fill_gaps(trace: np.ndarray) -> tuple[np.ndarray, float]:
    found = np.isfinite(trace)
    if found.sum() < 2:
        raise ValueError("no trace detected in image")
    idx = np.arange(len(trace))
    filled = trace.copy()
    filled[~found] = np.interp(idx[~found], idx[found], trace[found])
    return filled, float(found.mean())


def digitize_image(
    path: str | Path,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    px_per_mm: int = 8,
    lead_name: str = "II",
    dark_threshold: int = 110,
) -> SignalRepresentation:
    """Recover a single-lead ``SignalRepresentation`` from a gridded strip (TZ §4)."""
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    height, width = gray.shape
    px_per_s = px_per_mm * mm_per_s
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    trace, coverage = _fill_gaps(_trace_rows(gray < dark_threshold, 0, width, 0, height))
    signal_uv = (height // 2 - trace) * uv_per_px
    fs = float(px_per_s)
    return SignalRepresentation(
        signal=signal_uv.astype(np.float32).reshape(1, -1), sampling_rate_hz=fs,
        lead_names=[lead_name], duration_s=float(width / fs), source="image",
        provenance=Provenance(digitized=True, confidence=round(coverage, 3)),
    )


# ----------------------------------------------------------------------------
# 12-lead page (v2)
# ----------------------------------------------------------------------------
@dataclass
class PageGeometry:
    px_per_mm: int
    mm_per_s: float
    mm_per_mv: float
    width: int
    height: int
    margin: int
    row_h: int
    panel_w: int
    x0: int = 0          # left edge of the trace area (margin + pulse area)

    @property
    def px_per_s(self) -> float:
        return self.px_per_mm * self.mm_per_s

    @property
    def px_per_uv(self) -> float:
        return self.px_per_mm * self.mm_per_mv / 1000.0

    def panel_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        """(x0, x1, y0, y1) of a layout panel; row 3 = rhythm strip (full width)."""
        y0 = self.margin + row * self.row_h
        left = self.x0 or self.margin
        if row == 3:
            return left, left + 4 * self.panel_w, y0, y0 + self.row_h
        x0 = left + col * self.panel_w
        return x0, x0 + self.panel_w, y0, y0 + self.row_h


PULSE_AREA_MM = 8.0     # left of the traces when a calibration pulse is drawn
PULSE_WIDTH_MM = 5.0    # 200 ms at 25 mm/s
PULSE_HEIGHT_MV = 1.0


def page_geometry(px_per_mm: int = 6, mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
                  pulse: bool = False) -> PageGeometry:
    margin = int(PAGE_MARGIN_MM * px_per_mm)
    row_h = int(ROW_HEIGHT_MM * px_per_mm)
    panel_w = int(PANEL_SECONDS * mm_per_s * px_per_mm)
    x0 = margin + (int(PULSE_AREA_MM * px_per_mm) if pulse else 0)
    return PageGeometry(px_per_mm, mm_per_s, mm_per_mv, x0 + margin + 4 * panel_w,
                        2 * margin + 4 * row_h, margin, row_h, panel_w, x0)


def render_ecg_page(
    sig: SignalRepresentation, path: str | Path, *, px_per_mm: int = 6,
    mm_per_s: float = 25.0, mm_per_mv: float = 10.0, calibration_pulse: bool = False,
) -> str:
    """Render a 12-lead 3x4 page with a lead-II rhythm strip (10 s of data needed).

    With ``calibration_pulse`` a 1 mV x 200 ms rectangle precedes each row, as on
    printed ECGs; ``digitize_page`` reads mm/mV from it.
    """
    import cv2

    g = page_geometry(px_per_mm, mm_per_s, mm_per_mv, pulse=calibration_pulse)
    img = np.full((g.height, g.width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
    if calibration_pulse:
        pw, ph = int(PULSE_WIDTH_MM * px_per_mm), int(PULSE_HEIGHT_MV * mm_per_mv * px_per_mm)
        for row in range(4):
            _x0, _x1, y0, y1 = g.panel_box(row, 0)
            base = (y0 + y1) // 2
            xa = g.margin + int(1.0 * px_per_mm)
            pts = np.array([[xa, base], [xa, base - ph], [xa + pw, base - ph], [xa + pw, base]], dtype=np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False, color=_TRACE, thickness=2)
    fs = sig.sampling_rate_hz
    seg = int(PANEL_SECONDS * fs)
    for row, leads in enumerate(LAYOUT_3X4):
        for col, lead in enumerate(leads):
            x0, _x1, y0, y1 = g.panel_box(row, col)
            data = sig.lead(lead)[col * seg:(col + 1) * seg]
            _draw_trace(img, data, fs, x0, (y0 + y1) // 2, g.px_per_s, g.px_per_uv)
            cv2.putText(img, lead, (x0 + int(LABEL_OFFSET_MM * px_per_mm), y0 + int(4 * px_per_mm)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.08 * px_per_mm, _TRACE, 1, cv2.LINE_AA)
    x0, _x1, y0, y1 = g.panel_box(3, 0)
    _draw_trace(img, sig.lead(RHYTHM_LEAD)[: int(4 * seg)], fs, x0, (y0 + y1) // 2, g.px_per_s, g.px_per_uv)
    cv2.putText(img, RHYTHM_LEAD, (x0 + int(LABEL_OFFSET_MM * px_per_mm), y0 + int(4 * px_per_mm)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.08 * px_per_mm, _TRACE, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)
    return str(path)


def degrade_image(
    src: str | Path, dst: str | Path, *, rotation_deg: float = 0.0, blur_px: int = 0,
    jpeg_quality: int | None = None, shadow: float = 0.0, salt_noise: float = 0.0, seed: int = 0,
    perspective: float = 0.0, vignette: float = 0.0, texture: float = 0.0,
) -> str:
    """Write a photo-like degraded copy.

    rotation, blur, JPEG, shadow gradient, salt noise, plus the phone-photo
    defects: ``perspective`` (corners displaced by up to this fraction of the
    size), ``vignette`` (radial darkening towards the corners), ``texture``
    (low-frequency paper/lighting unevenness).
    """
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if perspective > 0:
        d = perspective * np.array([w, h], dtype=np.float32)
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = src_pts + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * d
        dst_pts -= dst_pts.min(axis=0)
        hm = cv2.getPerspectiveTransform(src_pts, dst_pts)
        out_w, out_h = int(dst_pts[:, 0].max()) + 1, int(dst_pts[:, 1].max()) + 1
        img = cv2.warpPerspective(img, hm, (out_w, out_h), borderValue=(255, 255, 255))
        h, w = img.shape[:2]
    if rotation_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    if shadow > 0:
        ramp = np.linspace(1.0 - shadow, 1.0, w, dtype=np.float32)[None, :, None]
        img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    if vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        rr = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
        gain = (1.0 - vignette * rr ** 2).astype(np.float32)[:, :, None]
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if texture > 0:
        noise = rng.normal(0, 1, (h // 32 + 1, w // 32 + 1)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        gain = (1.0 + texture * noise / 3.0).astype(np.float32)[:, :, None]
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if blur_px and blur_px > 0:
        k = blur_px * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    if salt_noise > 0:
        mask = rng.random((h, w)) < salt_noise
        img[mask] = rng.choice([0, 255], size=(int(mask.sum()), 1))
    dst = str(dst)
    if jpeg_quality is not None:
        if not dst.lower().endswith((".jpg", ".jpeg")):
            dst = str(Path(dst).with_suffix(".jpg"))
        cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        cv2.imwrite(dst, img)
    return dst


def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle (degrees) from long near-horizontal grid/trace lines (Hough)."""
    import cv2

    edges = cv2.Canny(gray, 50, 150)
    min_len = max(50, gray.shape[1] // 3)
    # maxLineGap small: JPEG 8x8 block edges are axis-aligned and would otherwise
    # chain into long "lines" that outvote the real grid (skew read as 0.3 deg
    # instead of 3 deg on a q95 JPEG).
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=min_len, maxLineGap=5)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= 15:
            angles.append(ang)
    return float(np.median(angles)) if angles else 0.0


TRACE_RATIO = 0.6            # pixel darker than this fraction of the local mean = ink
MIN_COMPONENT_WIDTH = 0.05   # of the panel width: shorter blobs are labels / specks


def trace_mask(gray: np.ndarray, px_per_mm: int) -> np.ndarray:
    """Ink pixels, robust to shading and to the printed grid.

    Ratio to the local mean brightness (8 mm window) instead of a constant offset:
    the trace (~8 % of the paper brightness) passes under any shadow, the 5-mm grid
    lines (~75 %) never do.
    """
    import cv2

    k = 2 * int(4 * px_per_mm) + 1
    local = cv2.blur(gray.astype(np.float32), (k, k)) + 1.0
    # No morphological opening: it fragments the 2-px trace into blobs that the
    # component filter would then discard.
    return gray.astype(np.float32) < TRACE_RATIO * local


def _keep_long_components(sub: np.ndarray, min_width: int) -> np.ndarray:
    """Drop ink blobs narrower than ``min_width`` (lead labels, specks, grid stubs)."""
    import cv2

    n, labels, stats, _ = cv2.connectedComponentsWithStats(sub.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] >= min_width:
            keep[i] = True
    return keep[labels]


def _bands(profile: np.ndarray, n: int, min_gap: int) -> list[tuple[int, int]]:
    """The ``n`` widest runs of a projection profile above 10 % of its max."""
    thr = 0.10 * float(profile.max()) if profile.max() > 0 else 0.0
    on = profile > thr
    runs, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    merged: list[tuple[int, int]] = []
    for a, b in runs:  # close small gaps (letters, thin breaks)
        if merged and a - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    merged.sort(key=lambda r: r[1] - r[0], reverse=True)
    return sorted(merged[:n])


GRID_PITCH_RANGE_PX = (3, 64)   # periods searched (px): 1-mm pitch up to ~12 px, its 5-mm line period up to 64


GRID_TILE_PX = 256           # pitch is estimated per tile (a residual tilt smears a whole-page projection)


def _ac_peaks(prof: np.ndarray) -> list[float]:
    """Prominent autocorrelation peaks (px, sub-pixel) of a de-trended 1-D profile."""
    import cv2

    prof = prof.astype(np.float32)
    if prof.size < 3 * GRID_PITCH_RANGE_PX[1]:
        return []
    trend = cv2.blur(prof.reshape(1, -1), (61, 1)).ravel()   # ksize = (width, height)
    x = prof - trend
    x -= x.mean()
    if not np.any(x):
        return []
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0] + 1e-9
    lo, hi = GRID_PITCH_RANGE_PX
    out = []
    for k in range(max(lo, 4), min(hi, len(ac) - 4)):
        if (ac[k] > 0.10 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]
                and ac[k] - max(ac[k - 3], ac[k + 3]) > 0.05):
            y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
            denom = (y0 - 2 * y1 + y2)
            out.append(float(k + (0.5 * (y0 - y2) / denom if denom != 0 else 0.0)))
    return out


def _vote_pitch(peaks: list[float]) -> float | None:
    """Grid pitch from the pooled autocorrelation peaks of many tiles.

    The 5-mm line period P (most robust to blur) is chosen among the periods
    >= 15 px by harmonic support: the true P is backed by peaks at the 1-mm
    harmonics P/5, 2P/5, 3P/5, 4P/5, while the sum/difference aliases (P +/- p)
    are not. Pitch = P / 5. Without any period >= 15 px the smallest supported
    small period is returned.
    """
    if len(peaks) < 3:
        return None
    arr = np.asarray(peaks)

    def count(c: float) -> int:
        return int(np.sum(np.abs(arr - c) <= 0.75))

    def refined(c: float) -> float:
        sel = arr[np.abs(arr - c) <= 0.75]
        return float(np.median(sel)) if sel.size else c

    big = [(float(c), count(c)) for c in range(15, GRID_PITCH_RANGE_PX[1] + 1) if count(c) >= 2]
    if big:
        thr = max(2, 0.3 * max(n for _, n in big))
        cands = [(n + sum(count(k * c / 5.0) for k in (1, 2, 3, 4)), c) for c, n in big if n >= thr]
        top = max(sc for sc, _ in cands)
        best = min(c for sc, c in cands if sc >= 0.8 * top)
        # 2P carries the same harmonics as P; a period at P/10 (the true 1-mm
        # pitch) betrays it. With no 1-mm support at all (blurred photo) a
        # period-P grid still shows a peak at P, so the smallest strong period wins.
        if best / 2 >= 15 and count(best / 10) >= 2 and count(best / 5) >= 2:
            best = best / 2
        elif count(best / 5) < 2:
            best = min(c for _, c in cands)
        return refined(best) / 5.0
    centres = np.arange(4, 15)
    counts = np.array([count(c) for c in centres])
    if counts.max() < 2:
        return None
    thr = max(2, 0.3 * counts.max())
    small = [float(c) for c, n in zip(centres, counts) if n >= thr]
    return refined(min(small)) if small else None


def estimate_grid_pitch_xy(gray: np.ndarray) -> tuple[float | None, float | None]:
    """(pitch along x, pitch along y) in px per 1 mm from the grid's periodicity:
    autocorrelation peaks of de-trended darkness and edge profiles of every tile,
    pooled per axis and voted (5-mm period / 5 preferred, 1-mm period as backup)."""
    dark = 255.0 - gray.astype(np.float32)
    gx = np.abs(np.diff(dark, axis=1, prepend=dark[:, :1]))
    gy = np.abs(np.diff(dark, axis=0, prepend=dark[:1, :]))
    h, w = dark.shape
    t = GRID_TILE_PX
    px_peaks: list[float] = []
    py_peaks: list[float] = []
    for y in range(0, max(1, h - t + 1), t):
        for x in range(0, max(1, w - t + 1), t):
            tile = (slice(y, min(h, y + t)), slice(x, min(w, x + t)))
            px_peaks += _ac_peaks(dark[tile].mean(axis=0)) + _ac_peaks(gx[tile].mean(axis=0))
            py_peaks += _ac_peaks(dark[tile].mean(axis=1)) + _ac_peaks(gy[tile].mean(axis=1))
    return _vote_pitch(px_peaks), _vote_pitch(py_peaks)


def estimate_grid_pitch(gray: np.ndarray) -> float | None:
    """Pixels per 1 mm (median of the x and y estimates), None when neither axis
    shows a clear period."""
    px, py = estimate_grid_pitch_xy(gray)
    if px is not None and py is not None:
        # Row profiles are polluted by the (horizontal) traces; trust the column
        # estimate when the two disagree by more than 10 %.
        return float(np.median([px, py])) if abs(px - py) <= 0.1 * px else px
    return px if px is not None else py


def detect_calibration_pulse(mask: np.ndarray, x0: int, y0: int, y1: int, px_per_mm: float) -> tuple[float, int] | None:
    """(pulse height px, right edge x) of a 1-mV rectangle at the row's left edge.

    The rectangle's TOP edge is a horizontal ink run about PULSE_WIDTH_MM wide — a
    QRS never draws one — with vertical edges at both ends reaching down to the
    baseline; the height is measured from the top run to the lower end of those edges.
    """
    span = int((PULSE_AREA_MM + PULSE_WIDTH_MM + 4) * px_per_mm)
    sub = mask[y0:y1, x0: x0 + span]
    if sub.size == 0:
        return None
    min_w = int(0.7 * PULSE_WIDTH_MM * px_per_mm)
    best = None
    for y in range(sub.shape[0]):
        cols = np.flatnonzero(sub[y])
        if cols.size < min_w:
            continue
        # longest contiguous run in this row
        breaks = np.flatnonzero(np.diff(cols) > 1)
        starts = np.r_[0, breaks + 1]; ends = np.r_[breaks, cols.size - 1]
        k = int(np.argmax(ends - starts))
        a, b = int(cols[starts[k]]), int(cols[ends[k]])
        if b - a + 1 >= min_w and (best is None or y < best[0]):
            best = (y, a, b)
    if best is None:
        return None
    y_top, a, b = best
    # Stroke thickness of the top edge (rows of ink below y_top at mid-run).
    mid = (a + b) // 2
    thick = 1
    while y_top + thick < sub.shape[0] and sub[y_top + thick, mid]:
        thick += 1
    edges = []
    for c in (a, a + 1, b - 1, b):
        rows = np.flatnonzero(sub[y_top:, c])
        if rows.size:
            edges.append(int(rows[-1]))
    if len(edges) < 2:
        return None
    # Height between the stroke centres of the top edge and the baseline.
    height = float(np.median(edges)) - thick + 1
    if height < 4 * px_per_mm:
        return None
    return height, x0 + b + 1


DEWARP_MIN_TILT_DEG = 0.3    # angle change across the page below this = no perspective (a rotation is not one)
DEWARP_MIN_CORR = 0.5        # ... and the angle must correlate with position (not just scatter)


def _vanishing_point(segments: np.ndarray, horizontal: bool) -> np.ndarray | None:
    """Vanishing point (homogeneous) of one family of grid lines, from how their
    angle changes with position across the page; None when they stay parallel."""
    if len(segments) < 8:
        return None
    x1, y1, x2, y2 = segments.T
    ang = np.arctan2(y2 - y1, x2 - x1)
    if not horizontal:
        ang = ang - np.sign(ang) * np.pi / 2  # vertical lines measured from the vertical
    pos = (y1 + y2) / 2 if horizontal else (x1 + x2) / 2
    a, b = np.polyfit(pos, ang, 1)
    p0, p1 = float(pos.min()), float(pos.max())
    if abs(np.degrees(a * (p1 - p0))) < DEWARP_MIN_TILT_DEG:
        return None
    # A perspective makes the angle change *systematically* with position; a
    # rotation or noisy segments give a slope without correlation.
    if pos.std() == 0 or ang.std() == 0 or abs(np.corrcoef(pos, ang)[0, 1]) < DEWARP_MIN_CORR:
        return None
    def line(p, theta):
        # homogeneous line through the family's mean point at position p with angle theta
        if horizontal:
            pt = np.array([float(np.mean((x1 + x2) / 2)), p, 1.0]); d = np.array([np.cos(theta), np.sin(theta), 0.0])
        else:
            pt = np.array([p, float(np.mean((y1 + y2) / 2)), 1.0]); d = np.array([-np.sin(theta), np.cos(theta), 0.0])
        return np.cross(pt, pt + d)
    return np.cross(line(p0, a * p0 + b), line(p1, a * p1 + b))


def _grid_segments(gray: np.ndarray) -> np.ndarray | None:
    import cv2

    edges = cv2.Canny(gray, 40, 120)
    w = gray.shape[1]
    segs = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=80, minLineLength=max(40, w // 5), maxLineGap=15)
    return None if segs is None else np.asarray(segs, dtype=np.float64).reshape(-1, 4)


def _line_angle_spread(gray: np.ndarray) -> float:
    """How far the grid's line families are from parallel (degrees): the spread of
    the near-horizontal and near-vertical segment angles. Smaller = flatter page."""
    segs = _grid_segments(gray)
    if segs is None or len(segs) < 8:
        return float("inf")
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    h = ang[np.abs(ang) < 10]
    v = np.abs(ang[np.abs(np.abs(ang) - 90) < 10]) - 90
    parts = [np.std(a) for a in (h, v) if len(a) >= 4]
    return float(np.mean(parts)) if parts else float("inf")


def dewarp_page(gray: np.ndarray) -> np.ndarray:
    """Undo a mild perspective: the grid's horizontal and vertical line families
    converge to vanishing points under perspective; the homography that sends their
    vanishing line back to infinity rectifies the page (residual rotation/scale is
    handled by the skew and grid-pitch steps). Returns the input unchanged when the
    line families are parallel (no perspective)."""
    import cv2

    h, w = gray.shape
    segs = _grid_segments(gray)
    if segs is None:
        return gray
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    horiz = segs[np.abs(ang) < 10]
    vert = segs[np.abs(np.abs(ang) - 90) < 10]
    vh = _vanishing_point(horiz, True)
    vv = _vanishing_point(vert, False)
    if vh is None and vv is None:
        return gray
    if vh is None:
        vh = np.array([1.0, 0.0, 0.0])
    if vv is None:
        vv = np.array([0.0, 1.0, 0.0])
    linf = np.cross(vh, vv)
    if abs(linf[2]) < 1e-12:
        return gray
    linf = linf / linf[2]
    hm = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [linf[0], linf[1], 1.0]])
    # Keep the page centre fixed and its scale ~unchanged: normalise by the centre's w.
    c = hm @ np.array([w / 2, h / 2, 1.0])
    if not np.isfinite(c).all() or abs(c[2]) < 1e-6:
        return gray
    scale = np.array([[1 / c[2], 0, 0], [0, 1 / c[2], 0], [0, 0, 1]])
    hm = scale @ hm
    # Re-centre so the page centre stays put.
    c2 = hm @ np.array([w / 2, h / 2, 1.0]); c2 = c2 / c2[2]
    shift = np.array([[1, 0, w / 2 - c2[0]], [0, 1, h / 2 - c2[1]], [0, 0, 1]])
    hm = shift @ hm
    warped = cv2.warpPerspective(gray, hm, (w, h), borderValue=255)
    # Self-check: keep the rectified page only if its grid lines became clearly
    # more parallel (a rotation or a noisy fit must not trigger a warp).
    if not _line_angle_spread(warped) < _line_angle_spread(gray):
        return gray
    # The projective step leaves an affine residue (shear, anisotropic scale): make
    # the horizontal family horizontal and the vertical family vertical, then fix
    # the aspect from the grid squares (pitch along x vs y).
    segs2 = _grid_segments(warped)
    if segs2 is not None and len(segs2) >= 8:
        a2 = np.arctan2(segs2[:, 3] - segs2[:, 1], segs2[:, 2] - segs2[:, 0])
        hh = a2[np.abs(np.degrees(a2)) < 10]
        vv = a2[np.abs(np.abs(np.degrees(a2)) - 90) < 10]
        if len(hh) >= 4 and len(vv) >= 4:
            th = float(np.median(hh))
            vv = np.where(vv < 0, vv + np.pi, vv)          # vertical direction, pointing down
            tv = float(np.median(vv))
            basis = np.array([[np.cos(th), np.cos(tv)], [np.sin(th), np.sin(tv)]])
            if abs(np.linalg.det(basis)) > 0.5:
                a_inv = np.linalg.inv(basis)
                c = np.array([w / 2, h / 2])
                m = np.hstack([a_inv, (c - a_inv @ c).reshape(2, 1)])
                warped = cv2.warpAffine(warped, m, (w, h), borderValue=255)
    px, py = estimate_grid_pitch_xy(warped)
    if px and py and 0.8 < px / py < 1.25 and abs(px / py - 1.0) > 0.01:
        m = np.array([[1.0, 0.0, 0.0], [0.0, px / py, h / 2 * (1 - px / py)]])
        warped = cv2.warpAffine(warped, m, (w, h), borderValue=255)
    return warped


def digitize_page(
    path: str | Path, *, px_per_mm: float | None = None, mm_per_s: float = 25.0,
    mm_per_mv: float | None = None, dewarp: bool = True,
) -> dict[str, np.ndarray]:
    """Recover each lead's µV trace (at px_per_s Hz) from a 3x4 + rhythm-strip page.

    ``px_per_mm`` None -> estimated from the grid pitch (fallback 6); ``mm_per_mv``
    None -> read from the calibration pulse when present (fallback 10). Returns
    ``{lead: trace_uv}`` (12 panel leads at 2.5 s, plus ``"II_rhythm"`` at 10 s);
    ``digitize_page.last_coverage`` / ``last_calibration`` hold the provenance.
    """
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if dewarp:
        gray = dewarp_page(gray)
    skew = estimate_skew(gray)
    if abs(skew) > 0.2:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    calib: dict = {"px_per_mm_given": px_per_mm, "mm_per_mv_given": mm_per_mv}
    if px_per_mm is None:
        est = estimate_grid_pitch(gray)
        px_per_mm = est if est is not None else 6.0
        calib["px_per_mm_estimated"] = None if est is None else round(est, 2)
    def layout(ppm: float):
        ppm_i = max(1, int(round(ppm)))
        geo = page_geometry(ppm_i, mm_per_s, mm_per_mv or 10.0)
        # One component-filtered ink mask for everything: JPEG ringing, salt noise
        # and lead labels are short blobs; a single stray pixel at the page edge
        # would otherwise stretch the row/column extents to the whole image.
        mk = _keep_long_components(trace_mask(gray, ppm_i), int(MIN_COMPONENT_WIDTH * geo.panel_w))
        # Baselines from the horizontal ink projection (ink piles up on the flat
        # segments), panel boxes from the layout geometry around them: a band cut
        # at a profile threshold would clip the QRS/T excursions. Fall back to geometry.
        bd = _bands(mk.sum(axis=1).astype(float), 4, min_gap=max(int(3 * ppm_i), int(0.02 * gray.shape[0])))
        return ppm_i, geo, mk, bd

    px_per_mm_i, g, mask, bands = layout(px_per_mm)
    if len(bands) != 4 and calib.get("px_per_mm_given") is None:
        for alt in (px_per_mm * 2, px_per_mm / 2, 6.0):
            ppm_i2, g2, mask2, bands2 = layout(alt)
            if len(bands2) == 4:
                px_per_mm, px_per_mm_i, g, mask, bands = alt, ppm_i2, g2, mask2, bands2
                break
    if len(bands) == 4 and calib.get("px_per_mm_given") is None:
        # The layout itself is a second scale reference: rows are ROW_HEIGHT_MM
        # apart. When the grid estimate disagrees (blur leaves only the 5-mm lines
        # and P vs 2P is ambiguous), take the grid harmonic nearest the row pitch,
        # or the row pitch itself.
        centres = np.array([(a + b) / 2 for a, b in bands])
        ppm_rows = float(np.median(np.diff(centres))) / ROW_HEIGHT_MM
        calib["px_per_mm_from_rows"] = round(ppm_rows, 2)
        # The row pitch is the primary scale (it is what the panel geometry is
        # built from); the grid estimate refines it only when the two agree.
        new_ppm = (px_per_mm + ppm_rows) / 2 if abs(px_per_mm - ppm_rows) <= 0.05 * ppm_rows else ppm_rows
        if abs(new_ppm - px_per_mm) > 1e-6:
            px_per_mm = new_ppm
            calib["px_per_mm_estimated"] = round(px_per_mm, 2)
            px_per_mm_i, g, mask, bands = layout(px_per_mm)
    half = g.row_h // 2
    if len(bands) == 4:
        rows = [(max(0, (a + b) // 2 - half), min(gray.shape[0], (a + b) // 2 + half)) for a, b in bands]
    else:
        rows = [(g.panel_box(r, 0)[2], g.panel_box(r, 0)[3]) for r in range(4)]
    # Column extent from the ink of the middle row, split into four equal panels.
    ink_x = np.flatnonzero(mask[rows[1][0]:rows[1][1]].any(axis=0))
    if ink_x.size > 4 * px_per_mm_i:
        xa, xb = int(ink_x[0]), int(ink_x[-1]) + 1
    else:
        xa, xb = g.margin, g.margin + 4 * g.panel_w
    # Calibration pulse at the left of the rows: 1 mV -> px, and the traces start after it.
    pulse = detect_calibration_pulse(mask, xa, rows[1][0], rows[1][1], px_per_mm)
    if pulse is not None:
        height_px, right = pulse
        if mm_per_mv is None:
            mm_per_mv = height_px / px_per_mm / PULSE_HEIGHT_MV
        calib["mm_per_mv_from_pulse"] = round(height_px / px_per_mm, 2)
        xa = right + int(1.5 * px_per_mm)
        ink_x = np.flatnonzero(mask[rows[1][0]:rows[1][1], xa:].any(axis=0))
        if ink_x.size:
            xa = xa + int(ink_x[0])
    mm_per_mv = mm_per_mv or 10.0
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    calib.update(px_per_mm=round(float(px_per_mm), 2), mm_per_mv=round(float(mm_per_mv), 2), pulse_found=pulse is not None)
    edges = np.linspace(xa, xb, 5).astype(int)
    cols = [(int(edges[c]), int(edges[c + 1])) for c in range(4)]
    out: dict[str, np.ndarray] = {}
    coverages = []
    for r, (y0, y1) in enumerate(rows[:3]):
        for c, (x0, x1) in enumerate(cols):
            lead = LAYOUT_3X4[r][c]
            trace, cov = _fill_gaps(_trace_rows(mask, x0, x1, y0, y1))
            base = float(np.median(trace))                  # panel baseline = median row
            out[lead] = ((base - trace) * uv_per_px).astype(np.float32)
            coverages.append(cov)
    y0, y1 = rows[3]
    trace, cov = _fill_gaps(_trace_rows(mask, xa, xb, y0, y1))
    out["II_rhythm"] = ((float(np.median(trace)) - trace) * uv_per_px).astype(np.float32)
    coverages.append(cov)
    digitize_page.last_coverage = float(np.mean(coverages))  # type: ignore[attr-defined]
    digitize_page.last_calibration = calib  # type: ignore[attr-defined]
    return out


def page_to_signal(
    traces: dict[str, np.ndarray], *, px_per_mm: float | None = None, mm_per_s: float = 25.0,
    mm_per_mv: float | None = None, coverage: float | None = None, calibration: dict | None = None,
) -> SignalRepresentation:
    """10-s 12-lead SignalRepresentation from page traces, with lead time windows."""
    calibration = calibration or getattr(digitize_page, "last_calibration", None) or {}
    px = px_per_mm if px_per_mm is not None else calibration.get("px_per_mm", 6.0)
    g = page_geometry(max(1, int(round(px))), mm_per_s, mm_per_mv or calibration.get("mm_per_mv", 10.0))
    fs = float(g.px_per_s)
    n = int(round(4 * PANEL_SECONDS * fs))
    seg = int(round(PANEL_SECONDS * fs))
    signal = np.zeros((12, n), dtype=np.float32)
    windows: dict[str, tuple[float, float]] = {}
    for row, leads in enumerate(LAYOUT_3X4):
        for col, lead in enumerate(leads):
            li = STANDARD_12_LEADS.index(lead)
            if lead == RHYTHM_LEAD and "II_rhythm" in traces:
                t = traces["II_rhythm"][:n]
                signal[li, : len(t)] = t
                windows[lead] = (0.0, 4 * PANEL_SECONDS)
                continue
            t = traces.get(lead)
            if t is None:
                continue
            t = t[:seg]
            signal[li, col * seg: col * seg + len(t)] = t
            windows[lead] = (col * PANEL_SECONDS, (col + 1) * PANEL_SECONDS)
    cov = coverage if coverage is not None else getattr(digitize_page, "last_coverage", None)
    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=list(STANDARD_12_LEADS),
        duration_s=float(n / fs), source="image",
        provenance=Provenance(digitized=True, confidence=None if cov is None else round(cov, 3),
                              lead_windows=windows,
                              notes=["3x4 page: each lead valid only inside its time window"]
                                    + ([f"calibration: {calibration}"] if calibration else [])),
    )


def digitize_page_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """Convenience: ``digitize_page`` + ``page_to_signal`` with shared calibration."""
    traces = digitize_page(path, **kw)
    kw.pop("dewarp", None)
    return page_to_signal(traces, coverage=getattr(digitize_page, "last_coverage", None),
                          calibration=getattr(digitize_page, "last_calibration", None), **kw)
