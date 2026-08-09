"""Base interface for video composer providers.

All video composer implementations must provide synchronous and asynchronous
compose methods and a health check that returns a :class:`src.core.result.Result`.
"""

from __future__ import annotations

import abc
from typing import Awaitable

from src.core.result import Result


class BaseVideoComposer(abc.ABC):
    """Abstract base class for video composition providers.

    The contract mirrors other provider abstractions in the project (STT,
    Vision, Timeline, Review, TTS).
    """

    @abc.abstractmethod
    def compose(self, request) -> Result["VideoComposeResult", "VideoComposeError"]:
        """Compose a video synchronously from a :class:`VideoComposeRequest`.

        Returns a ``Result`` wrapping either a :class:`VideoComposeResult` on
        success or a ``VideoComposeError`` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compose_async(
        self, request
    ) -> Awaitable[Result["VideoComposeResult", "VideoComposeError"]]:
        """Compose a video asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as ``compose``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "VideoComposeError"]:
        """Simple health check – returns ``Result.Ok(True)`` if the composer is ready.
        """
        raise NotImplementedError
