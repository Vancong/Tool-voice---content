"""Exception hierarchy for the video composer module.

All custom exceptions inherit from :class:`VideoComposeError` so the provider can
wrap them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class VideoComposeError(Exception):
    """Base class for all video composition errors."""


class FFmpegError(VideoComposeError):
    """Raised when the FFmpeg command fails or returns a non‑zero exit code."""


class ComposeValidationError(VideoComposeError):
    """Raised when input validation of the composition request fails."""

