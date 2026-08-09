"""Base interface for vision analysis providers.

All vision providers must implement synchronous and asynchronous analysis methods
as well as a health‑check that returns a :class:`src.core.result.Result`
indicating whether the underlying Gemini API is reachable.
"""

from __future__ import annotations

import abc
from typing import Awaitable

from src.core.result import Result


class BaseVisionAnalyzer(abc.ABC):
    """Abstract base class for vision analysis providers.

    The contract mirrors the one used by other provider modules (STT, Scene,
    Frame) so the pipeline can treat each step uniformly.
    """

    @abc.abstractmethod
    def analyze(
        self, frame_extraction_result: "VisionAnalysisResult"
    ) -> Result["VisionAnalysisResult", "VisionError"]:
        """Analyze frames synchronously.

        Returns a ``Result`` wrapping either a :class:`VisionAnalysisResult`
        on success or a :class:`VisionError` on failure.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def analyze_async(
        self, frame_extraction_result: "VisionAnalysisResult"
    ) -> Awaitable[Result["VisionAnalysisResult", "VisionError"]]:
        """Analyze frames asynchronously using the shared thread‑pool.

        The returned awaitable resolves to the same ``Result`` type as
        :meth:`analyze`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def health_check(self) -> Result[bool, "VisionError"]:
        """Verify that the Gemini Vision API can be contacted.

        Returns ``Result.Ok(True)`` on success, otherwise an error.
        """
        raise NotImplementedError
