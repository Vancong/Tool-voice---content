"""Base interface for scene detection providers.

All providers must implement synchronous and asynchronous detection methods as
well as a ``health_check`` that returns a :class:`src.core.result.Result`
indicating whether the underlying detection library is operational.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Awaitable

from src.core.result import Result


class BaseSceneDetector(abc.ABC):
    """Abstract base class for scene‑detection providers.

    Concrete implementations should be lightweight wrappers around a concrete
    detection library (e.g., `PySceneDetect`).  The contract mirrors the one used
    for the STT providers so that the rest of the pipeline can treat all
    providers uniformly.
    """

    @abc.abstractmethod
    def detect(self, video_path: Path) -> Result["SceneDetectionResult", "SceneDetectionError"]:
        """Detect scenes synchronously.

        Returns a ``Result`` wrapping either a :class:`SceneDetectionResult` on
        success or a :class:`SceneDetectionError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def detect_async(
        self, video_path: Path
    ) -> Awaitable[Result["SceneDetectionResult", "SceneDetectionError"]]:
        """Detect scenes asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as
        :meth:`detect`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "SceneDetectionError"]:
        """Verify that the underlying detection library can be instantiated.

        Returns ``Result.Ok(True)`` on success, otherwise an error.
        """
        raise NotImplementedError
