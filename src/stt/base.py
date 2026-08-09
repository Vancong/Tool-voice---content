# -*- coding: utf-8 -*-
"""
src/stt/base.py

Abstract base class for Speech-to-Text providers.
All providers must implement transcribe(), transcribe_async(), and health_check().
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Awaitable

from src.core.result import Result


class BaseSTT(abc.ABC):
    """Abstract base class for Speech-to-Text providers."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        progress_callback=None,
        cancel_token=None,
    ) -> Result:
        """Synchronously transcribe an audio file.

        Returns Result.Ok(STTResult) on success, Result.Err(exception) on failure.
        """
        ...

    @abc.abstractmethod
    def transcribe_async(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        progress_callback=None,
        cancel_token=None,
    ) -> Awaitable[Result]:
        """Asynchronously transcribe using the shared thread pool.

        Returns an awaitable resolving to the same Result type as transcribe().
        """
        ...

    @abc.abstractmethod
    def health_check(self) -> Result:
        """Verify that the STT engine is ready.

        Returns Result.Ok(True) on success, Result.Err(exception) on failure.
        """
        ...
