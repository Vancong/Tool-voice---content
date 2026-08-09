"""Exception hierarchy for the frame‑extraction module.

All custom exceptions inherit from :class:`FrameExtractionError` so that the
provider can wrap them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class FrameExtractionError(Exception):
    """Base class for all frame‑extraction related errors."""


class FrameReadError(FrameExtractionError):
    """Raised when a video frame cannot be read from the source file."""


class FrameWriteError(FrameExtractionError):
    """Raised when writing a JPEG image to disk fails."""

