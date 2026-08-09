"""Exceptions for the Gemini Web automation module.
"""

from __future__ import annotations

from src.review.exceptions import ReviewError, ReviewGenerationError


class GeminiWebError(ReviewError):
    """Base exception for all Gemini Web errors."""
    pass


class GeminiWebAuthError(GeminiWebError):
    """Raised when the session is invalid, expired, or user is not logged in."""
    pass


class GeminiWebNavigationError(GeminiWebError):
    """Raised when failing to navigate to Gemini web or network failure."""
    pass


class GeminiWebDOMError(GeminiWebError):
    """Raised when expected DOM elements (chat input, send button, response) are missing."""
    pass


class GeminiWebGenerationError(GeminiWebError, ReviewGenerationError):
    """Raised when Gemini Web fails during generation (busy, rate limit, blocked)."""
    pass


class GeminiWebTimeoutError(GeminiWebError):
    """Raised when waiting for Gemini Web response times out."""
    pass
