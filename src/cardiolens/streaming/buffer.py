"""Sliding-window buffer for streaming ECG (TZ §4).

Accepts pushed sample chunks and emits fixed-length windows that overlap by a
configurable hop — e.g. 10 s windows advancing every 5 s (50% overlap) — for
online processing. Absolute sample indexing keeps window timestamps stable
regardless of chunk sizes; consumed samples are trimmed so memory stays bounded.
"""

from __future__ import annotations

import numpy as np


class SlidingWindower:
    """Emit overlapping fixed-length windows from an incremental sample stream."""

    def __init__(self, n_leads: int, sampling_rate_hz: float,
                 window_s: float = 10.0, hop_s: float = 5.0):
        if hop_s <= 0 or window_s <= 0 or hop_s > window_s:
            raise ValueError("require 0 < hop_s <= window_s")
        self.n_leads = n_leads
        self.fs = float(sampling_rate_hz)
        self.window_n = int(round(window_s * self.fs))
        self.hop_n = int(round(hop_s * self.fs))
        self._buf = np.empty((n_leads, 0), dtype=np.float32)
        self._buf_start = 0      # absolute index of buffer's first sample
        self._next_win_start = 0  # absolute index of the next window to emit

    def push(self, chunk: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Append a ``[n_leads, k]`` chunk; return any newly-complete windows.

        Each returned item is ``(start_sample, window)`` where ``window`` is a
        ``[n_leads, window_n]`` array and ``start_sample`` its absolute index.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_leads:
            raise ValueError(f"chunk must be [{self.n_leads}, k]")
        self._buf = np.concatenate([self._buf, chunk], axis=1)
        buf_end = self._buf_start + self._buf.shape[1]

        windows: list[tuple[int, np.ndarray]] = []
        while self._next_win_start + self.window_n <= buf_end:
            a = self._next_win_start - self._buf_start
            windows.append((self._next_win_start,
                            self._buf[:, a:a + self.window_n].copy()))
            self._next_win_start += self.hop_n
            # Trim samples no longer needed by any future window.
            drop = self._next_win_start - self._buf_start
            if drop > 0:
                self._buf = self._buf[:, drop:]
                self._buf_start += drop
        return windows

    @property
    def buffered_samples(self) -> int:
        return int(self._buf.shape[1])
