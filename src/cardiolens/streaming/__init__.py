"""Streaming / online ECG processing (TZ §4)."""

from cardiolens.streaming.buffer import SlidingWindower
from cardiolens.streaming.processor import StreamProcessor, WindowResult

__all__ = ["SlidingWindower", "StreamProcessor", "WindowResult"]
