"""Exception hierarchy for the vision analysis module.

All custom exceptions inherit from :class:`VisionError` so the provider can wrap
them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class VisionError(Exception):
    """Base class for all vision‑analysis related errors."""


class VisionAPIError(VisionError):
    """Raised when the Gemini API returns an error or an unexpected response."""


class ImageReadError(VisionError):
    """Raised when a frame image cannot be read from disk before sending to Gemini."""


class RateLimitError(VisionError):
    """Raised when the Gemini API signals a rate‑limit condition."""

