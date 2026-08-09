"""Domain models for Gemini Web automation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class SessionStatus(str, Enum):
    NOT_LOGGED_IN = "not_logged_in"
    LOGGED_IN = "logged_in"
    EXPIRED = "expired"


@dataclass
class GeminiWebConfig:
    session_file: Path = field(default_factory=lambda: Path("data/session/gemini.json"))
    user_data_dir: Path = field(default_factory=lambda: Path("data/session/browser_profile"))
    headless: bool = False
    timeout_ms: int = 120_000
    navigation_timeout_ms: int = 60_000
    typing_delay_ms: int = 5
    base_url: str = "https://gemini.google.com/app"
    login_url: str = "https://accounts.google.com/ServiceLogin?service=wise&continue=https%3A%2F%2Fgemini.google.com%2Fapp"


@dataclass(frozen=True)
class GeminiWebResponse:
    text: str
    raw_html: str = ""
    processing_time: float = 0.0
    model_name: str = "gemini-web-chat"
