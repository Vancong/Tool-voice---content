"""Exception hierarchy for the review generation module.

All custom exceptions inherit from :class:`ReviewError` so the provider can wrap
them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class ReviewError(Exception):
    """Base class for all review‑generation related errors."""


class PromptError(ReviewError):
    """Raised when the generated prompt is invalid or cannot be rendered."""


class ReviewGenerationError(ReviewError):
    """Raised when the API returns an error or an unexpected response."""


class ReviewAPIError(ReviewError):
    """Raised when the LLM API returns an HTTP or status error."""


class ReviewRateLimitError(ReviewError):
    """Raised when rate limit is exceeded."""


class ReviewNetworkError(ReviewError):
    """Raised when a network timeout or connection failure occurs."""


class ReviewParseError(ReviewError):
    """Raised when the LLM response cannot be parsed into structured format."""


class ReviewValidationError(ReviewError):
    """Raised when generated review content fails quality validation."""

