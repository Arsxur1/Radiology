"""Paper-ECG digitisation (TZ §4).

A basic OpenCV pipeline that recovers a single-lead trace from a clean gridded
image (scan / PDF render). This is deliberately the MVP scope from TZ §4:
"надёжно — чистые сканы и PDF; на грязных фото работать, но с низкой
provenance.confidence". Robust segmentation of dirty photos and multi-lead
layouts (nnU-Net-style) is v2 (TZ §17).

Calibration (px/mm, mm/s, mm/mV) is passed in for the MVP; automatic grid /
calibration-pulse detection is a documented TODO. ``provenance.confidence``
reflects the fraction of columns where a trace pixel was found, so a degraded
image yields a low-confidence result rather than a fabricated one (TZ §0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cardiolens.contracts.signal import Provenance, SignalRepresentation


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
    """Render one lead to a clean gridded PNG (a synthetic 'paper' ECG).

    Light grey grid at 1 mm / 5 mm; dark trace centred vertically. Used for
    round-trip testing of the digitiser and as a stand-in for clean scans.
    """
    import cv2

    sig = np.asarray(signal_uv, dtype=np.float64)
    px_per_s = px_per_mm * mm_per_s
    px_per_uv = px_per_mm * mm_per_mv / 1000.0

    n = len(sig)
    width = max(1, int(round(n / sampling_rate_hz * px_per_s)))
    height = int(round(30 * mm_per_mv / 10.0 * px_per_mm))  # ~30 mm tall
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    small = px_per_mm
    large = px_per_mm * 5
    for x in range(0, width, small):
        cv2.line(img, (x, 0), (x, height), (230, 220, 220), 1)
    for y in range(0, height, small):
        cv2.line(img, (0, y), (width, y), (230, 220, 220), 1)
    for x in range(0, width, large):
        cv2.line(img, (x, 0), (x, height), (200, 180, 180), 1)
    for y in range(0, height, large):
        cv2.line(img, (0, y), (width, y), (200, 180, 180), 1)

    center = height // 2
    xs = (np.arange(n) / sampling_rate_hz * px_per_s).astype(int)
    ys = (center - sig * px_per_uv).astype(int)
    ys = np.clip(ys, 0, height - 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=(20, 20, 20), thickness=2)

    path = str(path)
    cv2.imwrite(path, img)
    return path


# Degraded-image switch and adaptive mask, tuned on ecg-image-kit renders (DECISIONS
# D15y): clean crops have a dark fraction p90 0.07, augmented ones median 0.19.
# Above 0.10 the adaptive mask lifts median SNR -3.9 -> +1.5 dB on augmented and
# 7.3 -> 8.7 dB on random-grid-colour renders, with clean renders unchanged.
_DEGRADED_DARK_FRACTION = 0.10
_ADAPTIVE_RATIO = 0.70        # trace: darker than 70% of the local background
_ADAPTIVE_BG_MM = 3.0         # local-background window (median), in mm
# Degraded-path traces reached only ~1.5 dB median SNR, so they must not pass as
# interpretable at full confidence (pipeline escalates below
# quality.digitized_confidence_min).
_DEGRADED_CONFIDENCE_CAP = 0.5


def _adaptive_dark_mask(img: np.ndarray, px_per_mm: float) -> np.ndarray:
    """Pixels darker than their local (median-filtered) background."""
    import cv2

    smooth = cv2.GaussianBlur(img, (3, 3), 0)
    k = int(_ADAPTIVE_BG_MM * px_per_mm) | 1
    background = cv2.medianBlur(smooth, k).astype(np.float32)
    return smooth.astype(np.float32) < _ADAPTIVE_RATIO * background


def _track_trace(dark: np.ndarray) -> np.ndarray:
    """Row position of one continuous trace per column (NaN where none).

    Starts from the image centre; in each column picks the run of dark pixels
    containing, or nearest to, the previous position and takes its mean row.
    On ecg-image-kit renders this lifted median SNR from 9.1 to 12.7 dB
    versus per-column averaging of all dark pixels (DECISIONS D15y).
    """
    height, width = dark.shape
    trace = np.full(width, np.nan)
    prev = height / 2.0
    for x in range(width):
        rows = np.flatnonzero(dark[:, x])
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        dist = [0.0 if r[0] <= prev <= r[-1] else min(abs(r[0] - prev), abs(r[-1] - prev))
                for r in runs]
        prev = trace[x] = float(runs[int(np.argmin(dist))].mean())
    return trace


def digitize_image(
    path: str | Path | np.ndarray,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    px_per_mm: int = 8,
    lead_name: str = "II",
    dark_threshold: int = 110,
) -> SignalRepresentation:
    """Recover a single-lead ``SignalRepresentation`` from a gridded image (TZ §4).

    A pixel is trace only if it is dark in *every* colour channel, so a red or
    pink grid (dark in grayscale, bright in R) is rejected. Each column's dark
    pixels are split into contiguous runs and the run nearest the previous
    column's position is followed (``_track_trace``), so traces from
    neighbouring leads that enter the crop are not averaged in. Gaps are
    interpolated. Output sampling rate equals the horizontal resolution
    (px/mm × mm/s).
    """
    import cv2

    if isinstance(path, np.ndarray):          # an in-memory crop (digitize_page)
        img = path
    else:
        path = str(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"cannot read image: {path}")
    if img.ndim == 3:
        img = img[:, :, :3].max(axis=2)  # dark only if dark in all channels
    img = img.astype(np.uint8)
    height, width = img.shape

    px_per_s = px_per_mm * mm_per_s
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    center = height // 2

    dark = img < dark_threshold
    # A clean scan is mostly paper: the trace covers a few % of the pixels. Far
    # more "dark" means shading, creases or a dark grid, so compare each pixel to
    # its local background instead of a fixed level — and flag the image as
    # degraded (capped provenance confidence).
    degraded = bool(dark.mean() > _DEGRADED_DARK_FRACTION)
    if degraded:
        dark = _adaptive_dark_mask(img, px_per_mm)

    trace_px = _track_trace(dark)

    found = np.isfinite(trace_px)
    coverage = float(found.mean())
    confidence = min(coverage, _DEGRADED_CONFIDENCE_CAP) if degraded else coverage
    if found.sum() < 2:
        raise ValueError("no trace detected in image")
    # Interpolate gaps so the signal is continuous.
    idx = np.arange(width)
    trace_px[~found] = np.interp(idx[~found], idx[found], trace_px[found])

    signal_uv = (center - trace_px) * uv_per_px
    fs = float(px_per_s)
    signal = signal_uv.astype(np.float32).reshape(1, -1)

    return SignalRepresentation(
        signal=signal, sampling_rate_hz=fs, lead_names=[lead_name],
        duration_s=float(width / fs), source="image",
        provenance=Provenance(digitized=True, confidence=round(confidence, 3)),
    )


# --- Whole-page digitisation (D26) -------------------------------------------
# Standard 12-lead print: three rows of four 2.5-s leads plus a 10-s lead-II
# rhythm strip, each row starting with a 1 mV calibration pulse.
PAGE_LAYOUT_3X4_II = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"),
                      ("III", "aVF", "V3", "V6"), ("II",))
# The calibration pulse must measure 10 mm (±1.0) at the detected scale: on clean
# and random-grid-colour renders it read 10.28 mm whenever the scale was right,
# and 7.3-8.2 mm on degraded pages whose grid scale was wrong (D26).
_PULSE_MM_TOL = 1.0
_ROW_SPACING_CV_MAX = 0.20   # the four baselines are roughly evenly spaced


@dataclass
class DigitizedPage:
    """Leads recovered from a whole 12-lead page (TZ §4)."""

    rhythm: SignalRepresentation                  # 10-s lead-II strip
    leads: dict[str, SignalRepresentation]        # 2.5-s segment per lead
    px_per_mm: float
    notes: list[str] = field(default_factory=list)


def _grid_px_per_mm(img: np.ndarray) -> float:
    """Grid period from the projection spectrum of coloured (or light-grey) grid pixels."""
    mx = img.max(axis=2).astype(float)
    mn = img.min(axis=2).astype(float)
    grid = ((mx - mn) > 25) & (mx > 110)
    if grid.mean() < 0.005:                      # monochrome print: light-grey lines
        grid = (mx > 110) & (mx < 235)
    periods = []
    for prof in (grid.mean(axis=0), grid.mean(axis=1)):
        p = prof - prof.mean()
        n = len(p)
        ac = np.correlate(p, p, mode="full")[n - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        # Coarse period: first autocorrelation peak (thin 1-px lines have equally
        # strong FFT harmonics, so a raw FFT argmax can land on a harmonic).
        lags = [k for k in range(3, 61) if ac[k] > 0.25 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
        if not lags:
            continue
        p0 = float(lags[0])
        # Fine period: zero-padded spectrum within ±10% of the coarse one.
        f = np.abs(np.fft.rfft(p * np.hanning(n), n=8 * n))
        freqs = np.fft.rfftfreq(8 * n)
        band = (freqs >= 0.9 / p0) & (freqs <= 1.1 / p0)
        periods.append(1.0 / freqs[band][int(np.argmax(f[band]))])
    if not periods:
        raise ValueError("page layout not recognised: no grid found")
    per = float(np.median(periods))
    return per if per < 15 else per / 5          # only 5-mm lines visible: 5 mm period


def _row_baselines(dark: np.ndarray, px_per_mm: float, n_rows: int) -> list[int]:
    """Baselines = strongest peaks of the trace-pixel row profile (the isoelectric
    line dominates each row), at least 15 mm apart."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    w = dark.shape[1]
    prof = dark[:, int(0.10 * w):int(0.95 * w)].sum(axis=1).astype(float)
    prof = uniform_filter1d(prof, max(3, int(px_per_mm)))
    pk, props = find_peaks(prof, distance=int(15 * px_per_mm), prominence=prof.max() * 0.05)
    if len(pk) < n_rows:
        return sorted(int(y) for y in pk)
    top = pk[np.argsort(props["prominences"])[::-1][:n_rows]]
    return sorted(int(y) for y in top)


def _calibration_pulse(dark: np.ndarray, px_per_mm: float) -> tuple[int, float] | None:
    """(x of the pulse's falling edge, pulse height in mm) from the left margin."""
    xs, heights = [], []
    for x in range(int(0.08 * dark.shape[1])):
        ys = np.flatnonzero(dark[:, x])
        if ys.size == 0:
            continue
        for run in np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1):
            if 7 * px_per_mm <= len(run) <= 13 * px_per_mm:
                xs.append(x)
                heights.append(len(run) / px_per_mm)
    if not xs:
        return None
    # The pulse's edges share its height; a nearby QRS stroke of another height
    # must not move the edge (it would shift every lead box).
    med = float(np.median(heights))
    near = [x for x, hh in zip(xs, heights, strict=True) if abs(hh - med) <= 0.7]
    edge = max(near) if near else xs[int(np.argmin([abs(hh - med) for hh in heights]))]
    return edge, med


def _skew_angle(img: np.ndarray) -> float:
    """Page rotation (degrees, ±5) that makes the grid/trace row projection sharpest."""
    import cv2

    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA)
    ink = 255 - small
    h, w = small.shape
    best, best_angle = -1.0, 0.0
    for angle in np.arange(-5.0, 5.01, 0.25):
        rot = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
        prof = cv2.warpAffine(ink, rot, (w, h), borderValue=0).sum(axis=1).astype(float)
        score = float(np.var(np.diff(prof)))
        if score > best:
            best, best_angle = score, float(angle)
    return best_angle


def _flatten_shading(img: np.ndarray) -> np.ndarray:
    """Divide each channel by its large-scale (median-filtered) background."""
    import cv2

    out = np.empty_like(img)
    for c in range(3):
        bg = cv2.medianBlur(img[:, :, c], 51).astype(np.float32) + 1.0
        out[:, :, c] = np.clip(img[:, :, c].astype(np.float32) / bg * 235.0, 0, 255).astype(np.uint8)
    return out


def digitize_page(
    path: str | Path,
    *,
    mm_per_s: float = 25.0,
    mm_per_mv: float = 10.0,
    layout: tuple[tuple[str, ...], ...] = PAGE_LAYOUT_3X4_II,
    dark_threshold: int = 110,
) -> DigitizedPage:
    """Digitise a whole 12-lead page: find the grid scale, the rows and the leads.

    Self-checks refuse a page that cannot be read reliably (§0) instead of
    guessing: the calibration pulse must measure 10 mm at the detected scale,
    the rows must be evenly spaced and every lead box must lie on the page.
    Raises ValueError with the failed check; the caller can then fall back to
    per-lead crops with ``digitize_image``.
    """
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    # Photo corrections (D32): undo page rotation, and flatten shading on a
    # degraded image. Degradation is judged on the ORIGINAL image and any
    # correction caps provenance confidence below — flattening makes a shaded
    # photo look clean and must not switch off the escalation.
    corrections: list[str] = []
    degraded_orig = bool((img.max(axis=2) < dark_threshold).mean() > _DEGRADED_DARK_FRACTION)
    angle = _skew_angle(img)
    if abs(angle) >= 0.25:
        hh, ww = img.shape[:2]
        rot = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
        img = cv2.warpAffine(img, rot, (ww, hh), borderValue=(255, 255, 255))
        corrections.append(f"deskewed {angle:+.2f}°")
    if degraded_orig:
        img = _flatten_shading(img)
        corrections.append("shading flattened")
    h, w = img.shape[:2]
    dark = img.max(axis=2) < dark_threshold
    px = _grid_px_per_mm(img)
    pulse = _calibration_pulse(dark, px)
    if pulse is None:
        raise ValueError("page layout not recognised: no calibration pulse found")
    x0, pulse_mm = pulse
    expected_mm = 10.0 * mm_per_mv / 10.0
    if abs(pulse_mm - expected_mm) > _PULSE_MM_TOL:
        raise ValueError(f"page layout not recognised: calibration pulse {pulse_mm:.1f} mm "
                         f"at the detected scale (expected {expected_mm:.0f} mm)")
    base = _row_baselines(dark, px, len(layout))
    if len(base) != len(layout):
        raise ValueError(f"page layout not recognised: {len(base)} rows found, "
                         f"{len(layout)} expected")
    gaps = np.diff(base)
    if gaps.std() / gaps.mean() > _ROW_SPACING_CV_MAX:
        raise ValueError("page layout not recognised: rows are not evenly spaced")
    half = int(gaps.min() / 2)
    row_mm = 10.0 * mm_per_s                      # a row spans 10 s
    n_cols = max(len(r) for r in layout)
    seg = row_mm / n_cols * px
    if x0 + row_mm * px > w + 5 * px:
        raise ValueError("page layout not recognised: rows run past the page edge")

    leads: dict[str, SignalRepresentation] = {}
    rhythm = None
    for r, names in enumerate(layout):
        # Keep the band centred on the baseline (the digitiser's 0 mV): trim it
        # symmetrically where it would run off the page.
        hb = min(half, base[r], h - base[r])
        band = img[base[r] - hb:base[r] + hb]
        if len(names) == 1:                       # full-width rhythm strip
            crop = band[:, x0:min(w, int(round(x0 + row_mm * px)))]
            rhythm = digitize_image(crop, mm_per_s=mm_per_s, mm_per_mv=mm_per_mv,
                                    px_per_mm=px, lead_name=names[0])
            continue
        for k, name in enumerate(names):
            crop = band[:, int(round(x0 + k * seg)):int(round(x0 + (k + 1) * seg))]
            leads[name] = digitize_image(crop, mm_per_s=mm_per_s, mm_per_mv=mm_per_mv,
                                         px_per_mm=px, lead_name=name)
    if rhythm is None:
        raise ValueError("page layout has no rhythm strip")
    notes = [f"grid {px:.2f} px/mm, calibration pulse {pulse_mm:.1f} mm, rows at {base}"]
    if corrections:
        notes.append("photo corrections: " + ", ".join(corrections))
        for sr in [rhythm, *leads.values()]:
            sr.provenance.confidence = min(sr.provenance.confidence, _DEGRADED_CONFIDENCE_CAP)
    degraded = [n for n, s in leads.items() if s.provenance.confidence < 0.8]
    if rhythm.provenance.confidence < 0.8 or degraded:
        notes.append("degraded image: low provenance confidence (report escalates)")
    return DigitizedPage(rhythm=rhythm, leads=leads, px_per_mm=round(px, 3), notes=notes)
