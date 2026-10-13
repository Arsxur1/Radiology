"""Telegram collector agent: ECG photos from groups -> pipeline -> reviewer verdicts -> labelled dataset."""

from cardiolens.telegram.analyze import (
    analyse_photo,
    apply_verdict,
    feedback_from_verdict,
    format_card,
    is_ecg_photo,
    load_signal,
    parse_verdict,
    render_review_png,
    save_photo,
    save_signal,
)
from cardiolens.telegram.config import TelegramConfig, load_telegram_config
from cardiolens.telegram.store import DatasetStore, Record

__all__ = [
    "DatasetStore",
    "Record",
    "TelegramConfig",
    "analyse_photo",
    "apply_verdict",
    "feedback_from_verdict",
    "format_card",
    "is_ecg_photo",
    "load_signal",
    "load_telegram_config",
    "parse_verdict",
    "render_review_png",
    "save_photo",
    "save_signal",
]
