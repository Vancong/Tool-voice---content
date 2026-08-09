"""Exception hierarchy for the timeline builder module.

All custom exceptions inherit from :class:`TimelineError` so the provider can
wrap them in the shared :class:`src.core.result.Result` type.
"""

from __future__ import annotations


class TimelineError(Exception):
    """Base class for all timeline‑related errors."""


class TimelineBuildError(TimelineError):
    """Raised when the builder fails to construct the timeline due to unexpected data."""


class InvalidTimelineError(TimelineError):
    """Raised when the input ``VisionAnalysisResult`` is malformed or missing required fields."""

