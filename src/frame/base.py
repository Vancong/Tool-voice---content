"""Base interface for frame extraction providers.

All providers must implement synchronous and asynchronous extraction methods as
well as a ``health_check`` that returns a :class:`src.core.result.Result`
indicating whether the underlying extraction library (OpenCV) is operational.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Awaitable

from src.core.result import Result


class BaseFrameExtractor(abc.ABC):
    """Abstract base class for frame‑extraction providers.

    The contract mirrors the one used by other provider modules so the pipeline
    can treat every step uniformly.
    """

    @abc.abstractmethod
    def extract(
        self, scene_detection_result: "SceneDetectionResult", video_path: Path
    ) -> Result["FrameExtractionResult", "FrameExtractionError"]:
        """Extract frames synchronously.

        Returns a ``Result`` wrapping either a :class:`FrameExtractionResult`
        on success or a :class:`FrameExtractionError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def extract_async(
        self, scene_detection_result: "SceneDetectionResult", video_path: Path
    ) -> Awaitable[Result["FrameExtractionResult", "FrameExtractionError"]]:
        """Extract frames asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as
        :meth:`extract`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "FrameExtractionError"]:
        """Verify that OpenCV can be imported and a video can be opened.

        Returns ``Result.Ok(True)`` on success, otherwise an error.
        """
        raise NotImplementedError
