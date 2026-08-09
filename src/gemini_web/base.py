"""Base interface for Gemini Web automation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.core.result import Result
from src.gemini_web.exceptions import GeminiWebError
from src.gemini_web.models import GeminiWebResponse, SessionStatus
from src.review.models import ReviewResult
from src.timeline.models import TimelineResult


class BaseGeminiWeb(ABC):
    """Abstract interface for Gemini Web automation provider."""

    @abstractmethod
    def get_session_status(self) -> SessionStatus:
        """Check current authentication/session status."""
        pass

    @abstractmethod
    def login_interactive(self) -> Result[bool, GeminiWebError]:
        """Launch interactive browser for user to log into Google/Gemini."""
        pass

    @abstractmethod
    def clear_session(self) -> Result[bool, GeminiWebError]:
        """Clear saved browser session/cookies."""
        pass

    @abstractmethod
    def generate_review(
        self,
        timeline: TimelineResult,
        review_style: str = "documentary",
        language: str = "vi",
        target_duration: Optional[int] = None,
        custom_instructions: Optional[str] = None,
    ) -> Result[ReviewResult, GeminiWebError]:
        """Send prompt to Gemini Web chat and parse into ReviewResult."""
        pass
