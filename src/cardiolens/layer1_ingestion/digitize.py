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

    @property
    def px_per_s(self) -> float:
        return self.px_per_mm * self.mm_per_s

    @property
    def px_per_uv(self) -> float:
        return self.px_per_mm * self.mm_per_mv / 1000.0

    def panel_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        """(x0, x1, y0, y1) of a layout panel; row 3 = rhythm strip (full width)."""
        y0 = self.margin + row * self.row_h
        if row == 3:
            return self.margin, self.margin + 4 * self.panel_w, y0, y0 + self.row_h
        x0 = self.margin + col * self.panel_w
        return x0, x0 + self.panel_w, y0, y0 + self.row_h


def page_geometry(px_per_mm: int = 6, mm_per_s: float = 25.0, mm_per_mv: float = 10.0) -> PageGeometry:
    margin = int(PAGE_MARGIN_MM * px_per_mm)
    row_h = int(ROW_HEIGHT_MM * px_per_mm)
    panel_w = int(PANEL_SECONDS * mm_per_s * px_per_mm)
    return PageGeometry(px_per_mm, mm_per_s, mm_per_mv, 2 * margin + 4 * panel_w,
                        2 * margin + 4 * row_h, margin, row_h, panel_w)


def render_ecg_page(
    sig: SignalRepresentation, path: str | Path, *, px_per_mm: int = 6,
    mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
) -> str:
    """Render a 12-lead 3x4 page with a lead-II rhythm strip (10 s of data needed)."""
    import cv2

    g = page_geometry(px_per_mm, mm_per_s, mm_per_mv)
    img = np.full((g.height, g.width, 3), 255, dtype=np.uint8)
    _draw_grid(img, px_per_mm)
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
) -> str:
    """Write a photo-like degraded copy: rotation, blur, JPEG, shadow, salt noise."""
    import cv2

    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {src}")
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    if rotation_deg:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    if shadow > 0:
        ramp = np.linspace(1.0 - shadow, 1.0, w, dtype=np.float32)[None, :, None]
        img = np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=120, minLineLength=min_len, maxLineGap=20)
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


def digitize_page(
    path: str | Path, *, px_per_mm: int = 6, mm_per_s: float = 25.0, mm_per_mv: float = 10.0,
) -> dict[str, np.ndarray]:
    """Recover each lead's µV trace (at px_per_s Hz) from a 3x4 + rhythm-strip page.

    Returns ``{lead: trace_uv}`` (12 panel leads at 2.5 s, plus ``"II_rhythm"`` at
    10 s) and stores the mean coverage in ``digitize_page.last_coverage``.
    """
    import cv2

    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    skew = estimate_skew(gray)
    if abs(skew) > 0.2:
        h, w = gray.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), borderValue=255)
    g = page_geometry(px_per_mm, mm_per_s, mm_per_mv)
    # One component-filtered ink mask for everything: JPEG ringing, salt noise and
    # lead labels are short blobs; a single stray pixel at the page edge would
    # otherwise stretch the row/column extents to the whole image.
    mask = _keep_long_components(trace_mask(gray, px_per_mm), int(MIN_COMPONENT_WIDTH * g.panel_w))
    uv_per_px = 1.0 / g.px_per_uv

    # Baselines from the horizontal ink projection (ink piles up on the flat
    # segments), panel boxes from the layout geometry around them: a band cut at a
    # profile threshold would clip the QRS/T excursions. Fall back to geometry.
    bands = _bands(mask.sum(axis=1).astype(float), 4, min_gap=int(3 * px_per_mm))
    half = g.row_h // 2
    if len(bands) == 4:
        rows = [(max(0, (a + b) // 2 - half), min(gray.shape[0], (a + b) // 2 + half)) for a, b in bands]
    else:
        rows = [(g.panel_box(r, 0)[2], g.panel_box(r, 0)[3]) for r in range(4)]
    # Column extent from the ink of the middle row, split into four equal panels.
    ink_x = np.flatnonzero(mask[rows[1][0]:rows[1][1]].any(axis=0))
    if ink_x.size > 4 * px_per_mm:
        xa, xb = int(ink_x[0]), int(ink_x[-1]) + 1
    else:
        xa, xb = g.margin, g.margin + 4 * g.panel_w
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
    return out


def page_to_signal(
    traces: dict[str, np.ndarray], *, px_per_mm: int = 6, mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0, coverage: float | None = None,
) -> SignalRepresentation:
    """10-s 12-lead SignalRepresentation from page traces, with lead time windows."""
    g = page_geometry(px_per_mm, mm_per_s, mm_per_mv)
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
                              notes=["3x4 page: each lead valid only inside its time window"]),
    )


def digitize_page_to_signal(path: str | Path, **kw) -> SignalRepresentation:
    """Convenience: ``digitize_page`` + ``page_to_signal`` with shared calibration."""
    traces = digitize_page(path, **kw)
    return page_to_signal(traces, coverage=getattr(digitize_page, "last_coverage", None), **kw)
