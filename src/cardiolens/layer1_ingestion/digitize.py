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
    path: str | Path,
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

    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    if img.ndim == 3:
        img = img[:, :, :3].max(axis=2)  # dark only if dark in all channels
    height, width = img.shape

    px_per_s = px_per_mm * mm_per_s
    uv_per_px = 1000.0 / (px_per_mm * mm_per_mv)
    center = height // 2

    trace_px = _track_trace(img < dark_threshold)

    found = np.isfinite(trace_px)
    coverage = float(found.mean())
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
        provenance=Provenance(digitized=True, confidence=round(coverage, 3)),
    )
