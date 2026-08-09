"""Base interface for review generation providers.

All providers must implement synchronous and asynchronous generation methods as
well as a health‑check that returns a :class:`src.core.result.Result` indicating
whether the underlying LLM service (Gemini) is reachable.
"""

from __future__ import annotations

import abc
from typing import Awaitable

from src.core.result import Result


class BaseReviewGenerator(abc.ABC):
    """Abstract base class for review generation providers.

    Mirrors the provider pattern used across the project.
    """

    @abc.abstractmethod
    def generate(self, timeline_result) -> Result["ReviewResult", "ReviewError"]:
        """Generate a review synchronously from a :class:`TimelineResult`.

        Returns a ``Result`` wrapping either a :class:`ReviewResult` on success or a
        :class:`ReviewError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_async(
        self, timeline_result
    ) -> Awaitable[Result["ReviewResult", "ReviewError"]]:
        """Generate a review asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as ``generate``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "ReviewError"]:
        """Simple health check – returns ``Result.Ok(True)`` if the provider is ready.
        """
        raise NotImplementedError
