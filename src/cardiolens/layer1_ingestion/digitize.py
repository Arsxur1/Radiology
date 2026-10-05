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


def _track_trace(dark: np.ndarray, mu: float = 1.0, max_runs: int = 12) -> np.ndarray:
    """Row position of one continuous trace per column (NaN where none).

    Each column's dark pixels split into runs; the trace is the sequence of runs
    with the least total cost — vertical gap between consecutive runs (0 when they
    touch) plus ``mu`` × the run's distance from the crop centre (the lead's
    baseline band) — found by dynamic programming over the whole crop (D79). The
    earlier greedy tracker (nearest run to the previous column, D15y) jumped onto a
    neighbouring row's large QRS entering the band and stayed there: on clean scans
    of PM-ECG-ID the rhythm strip read at correlation 0.28.
    """
    height, width = dark.shape
    trace = np.full(width, np.nan)
    center = height / 2.0
    xs: list[int] = []
    cols: list[np.ndarray] = []                    # per visited column: (top, bottom, mean)
    for x in range(width):
        rows = np.flatnonzero(dark[:, x])
        if rows.size == 0:
            continue
        runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1)
        if len(runs) > max_runs:
            runs = sorted(runs, key=len, reverse=True)[:max_runs]
        xs.append(x)
        cols.append(np.array([(r[0], r[-1], r.mean()) for r in runs], dtype=float))
    if not xs:
        return trace
    cost = mu * np.abs(cols[0][:, 2] - center) / height
    back: list[np.ndarray] = []
    for i in range(1, len(xs)):
        p, c = cols[i - 1], cols[i]
        gap_cols = max(1.0, float(xs[i] - xs[i - 1]))
        # Vertical gap between run intervals [top, bottom]; 0 if they overlap or touch.
        gap = np.maximum(c[None, :, 0] - p[:, None, 1], p[:, None, 0] - c[None, :, 1])
        gap[gap <= 1] = 0.0
        tot = cost[:, None] + gap / gap_cols / height * 10.0
        back.append(np.argmin(tot, axis=0))
        cost = tot.min(axis=0) + mu * np.abs(c[:, 2] - center) / height
    j = int(np.argmin(cost))
    for i in range(len(xs) - 1, -1, -1):
        trace[xs[i]] = cols[i][j, 2]
        if i > 0:
            j = int(back[i - 1][j])
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
_PULSE_DARK_THRESHOLD = 170    # thin pulse lines on scans read lighter than the trace (D78)
_ROW_SPACING_CV_MAX = 0.20   # the four baselines are roughly evenly spaced
_ROW_MIN_GAP_MM = 8.0        # rows at least this far apart (D81; was 15)
_ROW_MIN_COVER = 0.35        # a trace row has ink within ±3 mm across >= 35% of the width (D81)
# Rhythm-strip span as the time axis (D57): accepted only when it is 10 s within
# _SPAN_TOL at the grid scale and starts at most _SPAN_START_MAX_MM after a left
# pulse — on noisy renders a looser gate let noise stretch the span.
_SPAN_TOL = 0.03
_SPAN_START_MAX_MM = 12.0


@dataclass
class DigitizedPage:
    """Leads recovered from a whole 12-lead page (TZ §4)."""

    rhythm: SignalRepresentation                  # 10-s lead-II strip
    leads: dict[str, SignalRepresentation]        # 2.5-s segment per lead
    px_per_mm: float
    notes: list[str] = field(default_factory=list)


def _five_mm_periods(profiles, w: int) -> list[tuple[float, float]]:
    """(px/mm, strength) per projection profile from the 5-mm grid-line period.

    The period is searched where a page can put it — px/mm in [W/420, W/170], i.e.
    5 mm in [W/84, W/34] — after removing slow shading; the strongest
    autocorrelation peak wins, halved when its half-lag is nearly as strong (a
    10-mm lag), then refined on a zero-padded spectrum (D57).
    """
    from scipy.ndimage import uniform_filter1d

    lo5, hi5 = w / 84.0, w / 34.0
    out = []
    for prof in profiles:
        n = len(prof)
        if n < 4 * hi5:
            continue
        p = prof - uniform_filter1d(prof, int(2 * hi5))
        ac = np.correlate(p, p, mode="full")[n - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        ks = [k for k in range(max(2, int(lo5)), int(hi5) + 1) if ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
        if not ks:
            continue
        k5 = max(ks, key=lambda k: ac[k])
        # A visible 1-mm grid makes every multiple of 1 mm a peak (4 mm can beat
        # 5 mm): when the 1-mm fundamental and its 5x peak are both present, the
        # scale is refined on the 1-mm line itself.
        p1 = next((k for k in range(3, int(w / 170.0) + 2)
                   if ac[k] > 0.2 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]), None)
        f = np.abs(np.fft.rfft(p * np.hanning(n), n=8 * n))
        freqs = np.fft.rfftfreq(8 * n)
        if p1 is not None and lo5 <= 5 * p1 <= hi5 and any(
                abs(k - 5 * p1) <= max(2, 0.04 * 5 * p1) for k in ks):
            band = (freqs >= 0.9 / p1) & (freqs <= 1.1 / p1)   # refine on the 1-mm line
            out.append((1.0 / freqs[band][int(np.argmax(f[band]))], float(ac[p1])))
            continue
        half = [k for k in ks if abs(k - k5 / 2) <= 2]
        if half and ac[half[0]] >= 0.6 * ac[k5]:
            k5 = half[0]
        if ac[k5] < 0.05:
            continue
        band = (freqs >= 0.93 / k5) & (freqs <= 1.07 / k5)
        out.append((1.0 / freqs[band][int(np.argmax(f[band]))] / 5.0, float(ac[k5])))
    return out


def _grid_px_per_mm(img: np.ndarray) -> float:
    """Grid period from the projection spectrum of coloured (or light-grey) grid pixels."""
    if max(img.shape[:2]) >= 2000:               # scans and photos (D57)
        mx = img.max(axis=2).astype(float)
        mn = img.min(axis=2).astype(float)
        grid = ((mx - mn) > 25) & (mx > 110)
        if grid.mean() < 0.005:
            grid = (mx > 110) & (mx < 235)
        w = max(img.shape[:2])
        ests = _five_mm_periods((grid.mean(axis=0), grid.mean(axis=1)), w)
        if len(ests) < 2:                        # faint or colour-shifted grid
            ink = 255.0 - mx
            ests += _five_mm_periods((ink.mean(axis=0), ink.mean(axis=1)), w)
        if not ests:
            raise ValueError("page layout not recognised: no grid found")
        return float(max(ests, key=lambda e: e[1])[0])
    mx = img.max(axis=2).astype(float)
    mn = img.min(axis=2).astype(float)
    grid = ((mx - mn) > 25) & (mx > 110)
    if grid.mean() < 0.005:                      # monochrome print: light-grey lines
        grid = (mx > 110) & (mx < 235)
    periods = []
    # Search lags up to ~W/30 px: on a 300-dpi scan or a 12-16 MP photo the 5-mm
    # lines sit 53-75 px apart, beyond a fixed 60-px window (D57).
    lag_max = max(61, int(max(img.shape[:2]) / 30))
    for prof in (grid.mean(axis=0), grid.mean(axis=1)):
        p = prof - prof.mean()
        n = len(p)
        ac = np.correlate(p, p, mode="full")[n - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        # Coarse period: first autocorrelation peak (thin 1-px lines have equally
        # strong FFT harmonics, so a raw FFT argmax can land on a harmonic).
        lags = [k for k in range(3, min(lag_max, n - 1))
                if ac[k] > 0.25 and ac[k] >= ac[k - 1] and ac[k] >= ac[k + 1]]
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


def _grid_box(img: np.ndarray, px_per_mm: float) -> tuple[int, int, int, int] | None:
    """(y0, y1, x0, x1) of the coloured grid area, widened by 3 mm; None without a
    coloured grid. Ink outside it (QR code, footer text, labels) is not trace (D78)."""
    mx = img.max(axis=2).astype(np.int16)
    mn = img.min(axis=2).astype(np.int16)
    grid = ((mx - mn) > 25) & (mx > 110)
    if grid.mean() < 0.02:
        return None
    out = []
    for prof in (grid.mean(axis=1), grid.mean(axis=0)):
        hi = float(np.percentile(prof, 90))
        idx = np.flatnonzero(prof >= 0.3 * hi)
        if hi <= 0 or idx.size == 0:
            return None
        pad = int(3 * px_per_mm)
        out += [max(0, int(idx[0]) - pad), min(len(prof), int(idx[-1]) + pad)]
    y0, y1, x0, x1 = out
    if (y1 - y0) < 0.3 * img.shape[0] or (x1 - x0) < 0.5 * img.shape[1]:
        return None
    return y0, y1, x0, x1


def _row_baselines(dark: np.ndarray, px_per_mm: float, n_rows: int) -> list[int]:
    """Baselines = strongest peaks of the trace-pixel row profile (the isoelectric
    line dominates each row), at least 8 mm apart, among peaks whose ±3 mm band has
    ink across most of the width: a trace row is a near-continuous line, a text line
    (footer, legend) is not. Some templates put the rhythm strip only 12 mm below the
    third row, which the former 15-mm spacing could not resolve (D81)."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    h, w = dark.shape
    x_lo, x_hi = int(0.10 * w), int(0.95 * w)
    prof = dark[:, x_lo:x_hi].sum(axis=1).astype(float)
    prof = uniform_filter1d(prof, max(3, int(px_per_mm)))
    pk, props = find_peaks(prof, distance=int(_ROW_MIN_GAP_MM * px_per_mm),
                           prominence=prof.max() * 0.05)
    b = max(1, int(3 * px_per_mm))
    cover = np.array([dark[max(0, y - b):min(h, y + b), x_lo:x_hi].any(axis=0).mean() for y in pk])
    keep = cover >= _ROW_MIN_COVER
    pk, prom = pk[keep], props["prominences"][keep]
    if len(pk) < n_rows:
        return sorted(int(y) for y in pk)
    top = pk[np.argsort(prom)[::-1][:n_rows]]
    return sorted(int(y) for y in top)


def _calibration_pulse(dark: np.ndarray, px_per_mm: float,
                       baselines: list[int] | None = None,
                       side: str = "left") -> tuple[int, float] | None:
    """(x of the pulse's inner edge, pulse height in mm).

    ``side="left"``: pulse in the left margin, the edge is its falling edge (the
    trace starts there). ``side="right"``: pulse after the end of each row, the edge
    is its rising edge (the row ends there) — some printers place it there (D57).
    With row ``baselines`` only runs whose bottom sits on a baseline (±2 mm) count:
    a QR code or label in the margin has runs of similar height (D57).
    """
    xs, heights = [], []
    w = dark.shape[1]
    cols = range(int(0.08 * w)) if side == "left" else range(int(0.55 * w), w)
    for x in cols:
        ys = np.flatnonzero(dark[:, x])
        if ys.size == 0:
            continue
        for run in np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1):
            # Bottom on a row baseline, or up to 10 mm above it: some templates draw
            # the pulse raised above the trace's isoelectric line (D78).
            if baselines and not any(-10 * px_per_mm <= int(run[-1]) - b <= 2 * px_per_mm
                                     for b in baselines):
                continue
            if 7 * px_per_mm <= len(run) <= 13 * px_per_mm:
                xs.append(x)
                heights.append(len(run) / px_per_mm)
    if not xs:
        return None
    # The pulse's edges share its height; a nearby QRS stroke of another height
    # must not move the edge (it would shift every lead box).
    med = float(np.median(heights))
    near = [x for x, hh in zip(xs, heights, strict=True) if abs(hh - med) <= 0.7]
    if not near:
        near = [xs[int(np.argmin([abs(hh - med) for hh in heights]))]]
    edge = max(near) if side == "left" else min(near)
    return edge, med


def _trace_span(dark: np.ndarray, baseline: int, px_per_mm: float, edge: int,
                side: str) -> tuple[int, int] | None:
    """(first, last) column of the trace along one row: columns with ink within
    ±3 mm of the baseline, joined across gaps up to 3 mm; the longest such stretch
    beyond the calibration pulse."""
    h, w = dark.shape
    b = int(3 * px_per_mm)
    band = dark[max(0, baseline - b):min(h, baseline + b)]
    cols = band.any(axis=0)
    lo, hi = (edge + 1, w) if side == "left" else (0, edge)
    xs = np.flatnonzero(cols[lo:hi]) + lo
    if xs.size == 0:
        return None
    runs = np.split(xs, np.flatnonzero(np.diff(xs) > 3 * px_per_mm) + 1)
    best = max(runs, key=lambda r: r[-1] - r[0])
    return int(best[0]), int(best[-1])


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
    # The calibration pulse is drawn with a thin line that a scanner renders lighter
    # than the trace: look for it with a softer threshold (D78).
    pulse_dark = img.max(axis=2) < max(dark_threshold, _PULSE_DARK_THRESHOLD)
    box = _grid_box(img, px)
    if box is not None:                           # ignore ink outside the grid (D78)
        y0, y1, x0, x1 = box
        keep = np.zeros_like(dark)
        keep[y0:y1, x0:x1] = True
        dark &= keep
        pulse_dark &= keep
    base = _row_baselines(dark, px, len(layout))
    rows_ok = base if len(base) == len(layout) else None
    expected_mm = 10.0 * mm_per_mv / 10.0
    row_mm = 10.0 * mm_per_s                      # a row spans 10 s
    def _find_pulse(mask: np.ndarray) -> tuple[tuple[int, float] | None, str]:
        found = _calibration_pulse(mask, px, rows_ok)
        if (found is None or abs(found[1] - expected_mm) > _PULSE_MM_TOL) and rows_ok:
            right = _calibration_pulse(mask, px, rows_ok, side="right")
            if right is not None and abs(right[1] - expected_mm) <= _PULSE_MM_TOL:
                return right, "right"
        return found, "left"

    pulse, side = _find_pulse(dark)
    if pulse is None or abs(pulse[1] - expected_mm) > _PULSE_MM_TOL:
        # Only then the softer threshold: where the pulse reads at the trace
        # threshold, the softer one shifts its edge (synthetic pages, D78).
        soft, soft_side = _find_pulse(pulse_dark)
        if soft is not None and abs(soft[1] - expected_mm) <= _PULSE_MM_TOL:
            pulse, side = soft, soft_side
    rhythm_row = next((r for r, names in enumerate(layout) if len(names) == 1), None)
    pulse_err = None
    x0, pulse_mm = 0, float("nan")
    if pulse is None:
        pulse_err = "no calibration pulse found"
    else:
        x0, pulse_mm = pulse
        if side == "right":                       # the row ends where the pulse starts
            x0 = int(round(x0 - row_mm * px))
            if x0 < 0:
                pulse_err = "rows run past the page edge"
        if pulse_err is None and abs(pulse_mm - expected_mm) > _PULSE_MM_TOL:
            pulse_err = (f"calibration pulse {pulse_mm:.1f} mm at the detected scale "
                         f"(expected {expected_mm:.0f} mm)")
    # No usable pulse (some templates print none): fall back to the rhythm strip,
    # whose trace spans exactly 10 s, and the standard gain — flagged, so the
    # report escalates (D78). Without a 10-s rhythm trace the page is refused.
    no_pulse_span = None
    if pulse_err is not None:
        if rows_ok and rhythm_row is not None:
            sp = _trace_span(dark, base[rhythm_row], px, -1, "left")
            if sp is not None and abs((sp[1] - sp[0]) / px - row_mm) <= _SPAN_TOL * row_mm:
                no_pulse_span = sp
        if no_pulse_span is None:
            raise ValueError(f"page layout not recognised: {pulse_err}")
        x0, side, pulse_mm = no_pulse_span[0], "left", float("nan")
        corrections.append(f"no usable calibration pulse ({pulse_err}): standard "
                           f"{mm_per_mv:.0f} mm/mV assumed, time axis from the rhythm strip")
    if len(base) != len(layout):
        raise ValueError(f"page layout not recognised: {len(base)} rows found, "
                         f"{len(layout)} expected")
    gaps = np.diff(base)
    # The lead rows are evenly spaced; a rhythm strip may sit closer (some templates
    # 12 mm below the third row) or farther, within 0.4-1.6 lead-row gaps (D81).
    lead_rows = [r for r, names in enumerate(layout) if len(names) > 1]
    lead_gaps = np.diff([base[r] for r in lead_rows]) if len(lead_rows) >= 3 else gaps
    other = [g for g in gaps if g not in lead_gaps] if len(lead_rows) >= 3 else []
    ref_gap = float(np.median(lead_gaps))
    if (lead_gaps.std() / lead_gaps.mean() > _ROW_SPACING_CV_MAX
            or any(not 0.4 * ref_gap <= g <= 1.6 * ref_gap for g in other)):
        raise ValueError("page layout not recognised: rows are not evenly spaced")
    # Band half-height from the lead-row spacing: a close rhythm strip must not clip
    # tall QRS of every row; foreign traces in the band are left to the tracker (D81).
    half = int(ref_gap / 2)
    n_cols = max(len(r) for r in layout)
    # Time axis from the rhythm strip itself: its trace spans exactly one row
    # (10 s). Some printers leave a gap between the pulse and the trace, and a
    # photo's horizontal scale can differ from the grid estimate by ~1%; both
    # shifted every lead box (D57).
    px_t = px
    if rhythm_row is not None:
        span = (no_pulse_span if no_pulse_span is not None
                else _trace_span(dark, base[rhythm_row], px, x0, side))
        if (span is not None and abs((span[1] - span[0]) / px - row_mm) <= _SPAN_TOL * row_mm
                and (side == "right" or 0 <= (span[0] - x0) / px <= _SPAN_START_MAX_MM)):
            x0, px_t = span[0], (span[1] - span[0]) / row_mm
            notes_span = f"time axis from rhythm-strip span ({px_t:.2f} px/mm horizontally)"
        else:
            notes_span = ""
    else:
        notes_span = ""
    seg = row_mm / n_cols * px_t
    if x0 + row_mm * px_t > w + 5 * px:
        raise ValueError("page layout not recognised: rows run past the page edge")

    def _crop(band: np.ndarray, a: float, b: float) -> np.ndarray:
        """Columns [a, b) resampled to the grid scale, so fs = px × mm/s holds."""
        import cv2

        c = band[:, int(round(a)):min(w, int(round(b)))]
        target = max(2, int(round((b - a) / px_t * px)))
        if c.shape[1] < 2 or target == c.shape[1]:
            return c
        return cv2.resize(c, (target, c.shape[0]), interpolation=cv2.INTER_NEAREST)

    leads: dict[str, SignalRepresentation] = {}
    rhythm = None
    for r, names in enumerate(layout):
        # Keep the band centred on the baseline (the digitiser's 0 mV): trim it
        # symmetrically where it would run off the page.
        hb = min(half, base[r], h - base[r])
        band = img[base[r] - hb:base[r] + hb]
        if len(names) == 1:                       # full-width rhythm strip
            crop = _crop(band, x0, x0 + row_mm * px_t)
            rhythm = digitize_image(crop, mm_per_s=mm_per_s, mm_per_mv=mm_per_mv,
                                    px_per_mm=px, lead_name=names[0])
            continue
        for k, name in enumerate(names):
            crop = _crop(band, x0 + k * seg, x0 + (k + 1) * seg)
            leads[name] = digitize_image(crop, mm_per_s=mm_per_s, mm_per_mv=mm_per_mv,
                                         px_per_mm=px, lead_name=name)
    if rhythm is None:
        raise ValueError("page layout has no rhythm strip")
    pulse_txt = (f"calibration pulse {pulse_mm:.1f} mm ({side})" if no_pulse_span is None
                 else "no calibration pulse")
    notes = [f"grid {px:.2f} px/mm, {pulse_txt}, rows at {base}"]
    if notes_span:
        notes.append(notes_span)
    if corrections:
        notes.append("photo corrections: " + ", ".join(corrections))
        for sr in [rhythm, *leads.values()]:
            sr.provenance.confidence = min(sr.provenance.confidence, _DEGRADED_CONFIDENCE_CAP)
    degraded = [n for n, s in leads.items() if s.provenance.confidence < 0.8]
    if rhythm.provenance.confidence < 0.8 or degraded:
        notes.append("degraded image: low provenance confidence (report escalates)")
    return DigitizedPage(rhythm=rhythm, leads=leads, px_per_mm=round(px, 3), notes=notes)
