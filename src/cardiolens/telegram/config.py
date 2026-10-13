"""Configuration of the Telegram collector agent (configs/telegram.yaml + env)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from cardiolens.config import _load_yaml, configs_dir


class TelegramConfig(BaseModel):
    api_id: int | None = None
    api_hash: str | None = None
    groups: list[str | int] = Field(default_factory=list)
    review_chat: str | int = "me"
    store_dir: str = "data/telegram"
    session: str = "data/telegram/session"
    backfill_limit: int = 0
    min_grid_pitch_px: float = 3.0
    mode: str = "A"
    narrative: bool = False
    feedback_path: str = "data/feedback/feedback.jsonl"


def load_telegram_config(path: str | Path | None = None) -> TelegramConfig:
    """YAML first, then ``TELEGRAM_API_ID`` / ``TELEGRAM_API_HASH`` from the environment."""
    p = Path(path) if path else configs_dir() / "telegram.yaml"
    raw = _load_yaml(p) if p.exists() else {}
    raw = dict(raw or {})
    api_id = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if api_id:
        raw["api_id"] = int(api_id)
    if api_hash:
        raw["api_hash"] = api_hash
    return TelegramConfig(**raw)
