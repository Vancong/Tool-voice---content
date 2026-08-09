"""Base interface for timeline builder providers.

All timeline builders must implement synchronous and asynchronous build methods as
well as a health‑check that returns a :class:`src.core.result.Result` indicating
whether the component is ready.
"""

from __future__ import annotations

import abc
from typing import Awaitable

from src.core.result import Result


class BaseTimelineBuilder(abc.ABC):
    """Abstract base class for timeline builders.

    The contract mirrors the other provider abstractions (STT, Scene, Frame).
    """

    @abc.abstractmethod
    def build(self, vision_analysis_result) -> Result["TimelineResult", "TimelineError"]:
        """Build a timeline synchronously from a :class:`VisionAnalysisResult`.

        Returns a ``Result`` wrapping either a :class:`TimelineResult` on success
        or a :class:`TimelineError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def build_async(
        self, vision_analysis_result
    ) -> Awaitable[Result["TimelineResult", "TimelineError"]]:
        """Build a timeline asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as ``build``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "TimelineError"]:
        """Simple health check – returns ``Result.Ok(True)`` if the builder is ready.
        """
        raise NotImplementedError
