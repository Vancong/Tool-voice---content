"""Exception hierarchy for the scene‑detection module.

All custom exceptions inherit from :class:`SceneDetectionError` so that the
provider can wrap them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class SceneDetectionError(Exception):
    """Base class for all scene‑detection related errors."""


class VideoOpenError(SceneDetectionError):
    """Raised when the video file cannot be opened or is unreadable."""


class SceneDetectError(SceneDetectionError):
    """Raised for generic failures of the underlying detection library."""

