"""Base interface for Text‑to‑Speech providers.

All TTS providers must implement synchronous and asynchronous synthesis methods
and expose a health‑check that returns a :class:`src.core.result.Result`.
"""

from __future__ import annotations

import abc
from typing import Awaitable

from src.core.result import Result


class BaseTTS(abc.ABC):
    """Abstract base class for TTS providers.

    Mirrors the provider pattern used across the project (STT, Vision, Timeline,
    Review).
    """

    @abc.abstractmethod
    def synthesize(self, review_result) -> Result["VoiceResult", "TTSError"]:
        """Synchronously synthesize speech from a :class:`ReviewResult`.

        Returns a ``Result`` wrapping either a :class:`VoiceResult` on success or a
        :class:`TTSError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def synthesize_async(
        self, review_result
    ) -> Awaitable[Result["VoiceResult", "TTSError"]]:
        """Asynchronously synthesize speech using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as ``synthesize``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "TTSError"]:
        """Simple health check – returns ``Result.Ok(True)`` if the provider is ready.
        """
        raise NotImplementedError
